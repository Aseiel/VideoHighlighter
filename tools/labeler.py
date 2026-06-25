import cv2
import os
import json
import re
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from datetime import datetime
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch

# NEW: YOLO imports for pose estimation
from ultralytics import YOLO

def natural_sort_key(path):
    """Natural sort key for human-friendly sorting (e.g., 8 before 10)"""
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    
    filename = path.stem
    return [convert(c) for c in re.split('([0-9]+)', filename)]

class VideoLabelerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 Video Labeler - AI-Powered Tracking")
        self.root.geometry("1200x800")
        
        # State
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.current_frame = 0
        self.points = {}
        self.occluded = set()  # keypoints explicitly marked hidden/inside this frame
        self.current_kp = 0
        self.labeled_frames = []
        self.is_playing = False
        self.slider_update = True
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.interpolation_mode = "tracker"
        
        # YOLO Tracker state
        self.yolo_model = None
        self.tracking_data = {}  # {frame: {keypoint: (x,y)}} - Stores successful coordinates
        self.tracking_active = False
        # Store history for prediction: {keypoint_name: [(frame, x, y), ...]}
        self.track_history = defaultdict(list) 
        self.tracking_method = "botsort"
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.5
        # Prediction & Smoothing state
        self.prediction_window_size = 3 # For extrapolating lost frames
        self.smoothing_window_size = 5  # Moving Average filter size (N frames)

        # Prediction variables
        self.prediction_window_size = 3 # How many frames back to use for prediction
        
        # Adjustment state
        self.adjusting_interpolated = False
        self.selected_interpolated_frame = None
        self.adjusting_keypoint = None
        
        # Spacebar state
        self.space_held = False
        
        # File navigation state
        self.video_files = []
        self.current_video_idx = 0
        self.progress_data = {}
        self.progress_file = Path("labeling_progress.json")
        
        # Track if current video has been exported
        self.current_video_exported = False
        
        # Keypoint definitions (user-defined, persisted to a config file)
        self.keypoint_config_file = Path("labeler_keypoints.json")
        self._keypoint_config_existed = self.keypoint_config_file.exists()
        self.keypoint_names = self._load_keypoint_names()
        self.keypoint_display_names = {}
        self.colors = {}      # BGR tuples for cv2 drawing
        self.color_hex = {}   # '#RRGGBB' for tk widgets
        self._rebuild_keypoint_meta()

        self.status_var = tk.StringVar(value="Ready. Select a keypoint, then click on the video to place it.")
        
        self.load_progress()
        self.setup_ui()
        self.bind_shortcuts()
        self._finalize_window()
        self.load_yolo_model()

        # First run (no config yet): nudge the user toward the inline editor
        if not self._keypoint_config_existed and not self.keypoint_names:
            self.status_var.set("No keypoints yet — add them in the 🎯 Keypoints panel (➕ Add Keypoint).")

    # *************************************************************
    # *** KEYPOINT LABEL CONFIG (user-defined) ***
    # *************************************************************
    DEFAULT_KEYPOINTS = []  # none shipped in code; defined by the user on first run

    def _load_keypoint_names(self):
        """Load keypoint names from the config file, falling back to defaults."""
        try:
            if self.keypoint_config_file.exists():
                with open(self.keypoint_config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                names = [str(n).strip() for n in data.get('keypoint_names', []) if str(n).strip()]
                if names:
                    return names
        except Exception as e:
            print(f"⚠️ Could not read {self.keypoint_config_file}: {e}")
        return list(self.DEFAULT_KEYPOINTS)

    def _save_keypoint_names(self):
        """Persist the current keypoint names to the config file."""
        try:
            with open(self.keypoint_config_file, 'w', encoding='utf-8') as f:
                json.dump({'keypoint_names': self.keypoint_names}, f, indent=2)
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save keypoint config:\n{e}")

    def _rebuild_keypoint_meta(self):
        """Regenerate display names + evenly-spaced colors for the current keypoints."""
        import colorsys
        self.keypoint_display_names = {}
        self.colors = {}
        self.color_hex = {}
        n = max(len(self.keypoint_names), 1)
        for i, name in enumerate(self.keypoint_names):
            r, g, b = colorsys.hsv_to_rgb(i / n, 0.85, 1.0)
            R, G, B = int(r * 255), int(g * 255), int(b * 255)
            self.colors[name] = (B, G, R)  # cv2 uses BGR
            self.color_hex[name] = f'#{R:02X}{G:02X}{B:02X}'
            self.keypoint_display_names[name] = name.replace('_', ' ').title()

    def _add_keypoint(self):
        """Append a new keypoint with a unique default name; ready to rename."""
        existing = set(self.keypoint_names)
        i = 1
        while f"label{i}" in existing:
            i += 1
        new_name = f"label{i}"
        self.keypoint_names.append(new_name)
        self._rebuild_keypoint_meta()
        self._save_keypoint_names()
        self._populate_keypoint_buttons()
        self.select_keypoint(new_name)
        # Focus its name field so the user can type a real name immediately
        entry = self.kp_buttons.get(new_name, {}).get('entry')
        if entry:
            entry.focus_set()
            entry.select_range(0, tk.END)
        self.status_var.set(f"➕ Added '{new_name}' — type a name and press Enter")

    def _remove_keypoint(self, name):
        """Remove a keypoint and strip its points so data stays consistent."""
        if name not in self.keypoint_names:
            return
        self.keypoint_names.remove(name)
        for f in self.labeled_frames:
            f.get('points', {}).pop(name, None)
        self.points.pop(name, None)
        self._rebuild_keypoint_meta()
        self._save_keypoint_names()
        self._populate_keypoint_buttons()
        self.current_kp = 0
        if self.keypoint_names:
            self.select_keypoint(self.keypoint_names[0])
        if self.cap:
            self.show_frame(self.current_frame)
        self.status_var.set(f"🗑️ Removed keypoint '{name}'")

    def _commit_rename(self, old, new):
        """Rename a keypoint in place, migrating any existing labeled points."""
        new = (new or "").strip()
        if old not in self.keypoint_names or new == old:
            return
        if not new:
            self.kp_buttons[old]['name_var'].set(old)  # revert blank
            return
        if new in self.keypoint_names:
            self.status_var.set(f"⚠️ '{new}' already exists")
            self.kp_buttons[old]['name_var'].set(old)
            return
        self.keypoint_names[self.keypoint_names.index(old)] = new
        # Migrate points so renaming doesn't orphan existing labels
        for f in self.labeled_frames:
            pts = f.get('points', {})
            if old in pts:
                pts[new] = pts.pop(old)
        if old in self.points:
            self.points[new] = self.points.pop(old)
        self._rebuild_keypoint_meta()
        self._save_keypoint_names()
        self._populate_keypoint_buttons()
        self.select_keypoint(new)
        if self.cap:
            self.show_frame(self.current_frame)
        self.status_var.set(f"✏️ Renamed '{old}' → '{new}'")

    def _populate_keypoint_buttons(self):
        """(Re)build the inline keypoint rows: select ◉ + editable name + ✕,
        plus an ➕ Add Keypoint button. Lives in self.kp_container."""
        for child in self.kp_container.winfo_children():
            child.destroy()
        self.kp_buttons = {}

        for i, name in enumerate(self.keypoint_names):
            row = ttk.Frame(self.kp_container)
            row.pack(fill=tk.X, pady=2)

            rb = ttk.Radiobutton(row, variable=self.kp_select_var, value=name,
                                 command=lambda n=name: self.select_keypoint(n))
            rb.pack(side=tk.LEFT)

            color_canvas = tk.Canvas(row, width=16, height=16, bg='gray',
                                     highlightthickness=1)
            color_canvas.pack(side=tk.LEFT, padx=(0, 3))

            name_var = tk.StringVar(value=name)
            entry = ttk.Entry(row, textvariable=name_var, width=13)
            entry.pack(side=tk.LEFT)
            # Commit rename on Enter / focus-out (deferred so the widget isn't
            # destroyed while still handling its own event)
            entry.bind('<Return>', lambda e, o=name, v=name_var:
                       self.root.after_idle(lambda: self._commit_rename(o, v.get())))
            entry.bind('<FocusOut>', lambda e, o=name, v=name_var:
                       self.root.after_idle(lambda: self._commit_rename(o, v.get())))

            status_label = ttk.Label(row, text="⬜", foreground='gray', width=2)
            status_label.pack(side=tk.LEFT, padx=2)

            ttk.Button(row, text="✕", width=2,
                       command=lambda n=name: self._remove_keypoint(n)).pack(side=tk.RIGHT)

            self.kp_buttons[name] = {
                'radio': rb, 'name_var': name_var, 'entry': entry,
                'color': color_canvas, 'status': status_label, 'placed': False
            }

        ttk.Button(self.kp_container, text="➕ Add Keypoint",
                   command=self._add_keypoint).pack(anchor=tk.W, pady=(4, 0))

        if not self.keypoint_names:
            ttk.Label(self.kp_container, text="No labels yet — click ➕ Add Keypoint.",
                      foreground='gray', font=('Arial', 8)).pack(anchor=tk.W, pady=2)

        # Rebuild shortcut hints
        for child in self.kp_shortcut_container.winfo_children():
            child.destroy()
        for i, name in enumerate(self.keypoint_names[:9]):
            ttk.Label(self.kp_shortcut_container, text=f"{i+1}: {name[:8]}",
                      foreground='gray', font=('Arial', 8)).pack(side=tk.LEFT, padx=3)

    # *************************************************************
    # *** CORE SMOOTHING FUNCTIONALITY ***
    # *************************************************************
    def apply_moving_average(self):
        """Applies a moving average filter to all stored keypoint coordinates."""
        if not self.labeled_frames:
            return 0

        # We will modify the 'points' dictionary in place
        smoothed_count = 0
        
        for labeled in self.labeled_frames:
            frame = labeled['frame']
            original_points = labeled['points']
            new_points = {}
            
            for kp_name, (x, y) in original_points.items():
                # Collect coordinates from the last N frames for this specific keypoint
                coordinates = []
                start_idx = max(0, self.labeled_frames.index(labeled) - self.smoothing_window_size)
                end_idx = min(len(self.labeled_frames), self.labeled_frames.index(labeled) + 1)

                for i in range(start_idx, end_idx):
                    # Check if the point exists in the windowed frame
                    temp_label = self.labeled_frames[i]
                    if kp_name in temp_label['points']:
                        coords = temp_label['points'][kp_name]
                        coordinates.append((coords[0], coords[1]))

                # Calculate the average (the smoothed point)
                if coordinates:
                    avg_x = int(sum(c[0] for c in coordinates) / len(coordinates))
                    avg_y = int(sum(c[1] for c in coordinates) / len(coordinates))
                    new_points[kp_name] = (avg_x, avg_y)
                else:
                    # Should not happen if the keypoint existed originally
                    new_points[kp_name] = (x, y) 

            labeled['points'] = new_points
            smoothed_count += 1
        
        return smoothed_count

    # ============ YOLO MODEL LOADING ============
    
    def load_yolo_model(self):
        """Load YOLO model with progress feedback"""
        try:
            self.status_var.set("🔄 Loading YOLO model...")
            self.root.update()
            
            # Use YOLOv8n-pose for keypoint detection
            model_name = "yolov8n-pose.pt"  # Pose estimation model
            
            # Check if model exists locally, if not download
            model_path = Path(model_name)
            if not model_path.exists():
                self.status_var.set(f"📥 Downloading {model_name}... (first time only)")
                self.root.update()
            
            self.yolo_model = YOLO(model_name)
            self.status_var.set(f"✅ YOLO pose model loaded: {model_name}")
            
        except Exception as e:
            self.status_var.set(f"❌ Failed to load YOLO: {str(e)}")
            messagebox.showerror("YOLO Error", 
                f"Could not load YOLO model.\n\nError: {str(e)}\n\nMake sure to install ultralytics:\npip install ultralytics")
            self.yolo_model = None
    
    # ============ YOLO TRACKING METHODS ============
    
    def track_with_yolo(self, start_frame=None, end_frame=None, use_manual_anchors=True):
        """Enhanced tracking using YOLO pose estimation with prediction fallback."""
        if not self.cap:
            messagebox.showwarning("No Video", "Please load a video first")
            return False
        
        if self.yolo_model is None:
            messagebox.showwarning("No YOLO Model", "YOLO model not loaded. Please check installation.")
            return False
        
        # Determine tracking range
        if start_frame is None:
            start_frame = self.current_frame
        
        if end_frame is None or end_frame >= self.total_frames:
            end_frame = self.total_frames - 1
        
        # Collect manual frames for reference
        manual_frames = {}
        for labeled in self.labeled_frames:
            if labeled.get('manual', False):
                frame = labeled['frame']
                if start_frame <= frame <= end_frame:
                    manual_frames[frame] = labeled['points']
        
        # Determine which labels to track
        if hasattr(self, 'manual_labels_only') and self.manual_labels_only.get():
            manual_labels = set()
            for frame_points in manual_frames.values():
                manual_labels.update(frame_points.keys())
            if not manual_labels:
                messagebox.showwarning("No Manual Labels", 
                    "Please label at least one frame manually first.")
                return False
        else:
            manual_labels = set(self.keypoint_names)  # Track all
        
        # Reset tracking state for the new run
        self.tracking_active = True
        self.track_history = defaultdict(list) 
        
        self.status_var.set(f"🔄 YOLO Tracking frames {start_frame}-{end_frame}... (Press ESC to stop)")
        self.root.update()
        
        tracked_frames = []
        
        # Process video
        for frame_idx in range(start_frame, end_frame + 1):
            if not self.tracking_active:
                break
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_frame_points = {}
            temp_results = []

        # 1. Use manual points if available (Highest priority)
        if frame_idx in manual_frames:
            current_frame_points = manual_frames[frame_idx]
            temp_results = [{'points': current_frame_points, 'tracked': True}] 
        else:
            # Run YOLO pose estimation
            results = None
            try:
                results = self.yolo_model.track(
                    frame, 
                    persist=True,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
            except Exception as e:
                print(f"YOLO Error at frame {frame_idx}: {e}")
                results = None

                # Handle detection results
                if results and results[0].keypoints is not None:
                    keypoints_data = results[0].keypoints
                    
                    for i, (keypoints, box) in enumerate(zip(keypoints_data.xy, results[0].boxes.xyxy)):
                        temp_points = self._map_coco_to_keypoints(keypoints, frame.shape)
                        if not temp_points: continue

                        filtered_points = {}
                        for label in manual_labels:
                            if label in temp_points:
                                filtered_points[label] = temp_points[label]
                        
                        if filtered_points:
                            current_frame_points = filtered_points
                            temp_results.append({
                                'points': current_frame_points, 
                                'tracked': True, 
                                'track_id': int(results[0].boxes.id[i].item()) if results[0].boxes.id is not None and i < len(results[0].boxes.id) else None,
                                'conf': float(results[0].boxes.conf[i].item()) if results[0].boxes.conf is not None and i < len(results[0].boxes.conf) else 0.0
                            })
                    
                    if temp_results:
                        current_frame_points = temp_results[0]['points']

                # Handle detection failure -> use prediction fallback
                else:
                    print(f"⚠️ Warning: YOLO lost track at frame {frame_idx}. Using historical data.")
                    current_frame_points = self._predict_keypoints(start_frame, end_frame)


            # 2. Add the result to tracked_frames and update history/progress
            if current_frame_points:
                tracked_frames.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / self.fps,
                    'points': current_frame_points.copy(), 
                    'manual': False,
                    # We use a general check to see if *any* detection happened (even bad ones)
                    'tracked': bool(results and results[0].keypoints is not None), 
                    'track_id': temp_results[0].get('track_id') if temp_results else None,
                    'tracking_method': self.tracking_method,
                    'yolo_conf': 1.0 # Placeholder for prediction/manual frames
                })

                # Update history with the successfully calculated points
                for label in manual_labels:
                    if label in current_frame_points:
                        self.track_history[label].append((frame_idx, current_frame_points[label][0], current_frame_points[label][1]))
            
            # Update progress (Only update status if we processed a frame)
            if frame_idx % 20 == 0 and not (start_frame <= frame_idx < end_frame):
                progress = int((frame_idx - start_frame) / (end_frame - start_frame + 1) * 100)
                self.status_var.set(f"🔄 Tracking: {progress}% complete (frame {frame_idx}/{end_frame})")
                self.root.update()
        
        # Update overall history and label list
        self.labeled_frames = [f for f in self.labeled_frames if f.get('manual', False) or f.get('manual_correction', False)]
        self.labeled_frames.extend(tracked_frames)
        self.labeled_frames.sort(key=lambda x: x['frame'])
        
        # === APPLY SMOOTHING AFTER ALL FRAMES ARE POPULATED (FIXED LINE CALL) ===
        smoothed_count = self.apply_moving_average() 
        print(f"✨ Smoothing applied across {smoothed_count} points.") # User feedback
        
        self.progress_bar['value'] = len(self.labeled_frames)

        # Update UI
        self.progress_bar['value'] = len(self.labeled_frames)
        self.progress_label.config(text=f"{len(self.labeled_frames)} / {self.total_frames} frames")
        
        if self.video_path:
            self.save_frame_progress()
        
        self.tracking_active = False
        self.status_var.set(f"✅ YOLO Tracking complete! Added {len(tracked_frames)} frames (Prediction & Smoothing applied)")
        
        # Display current frame
        self.show_frame(self.current_frame)
        return True
    
    def _predict_keypoints(self, start_frame, end_frame):
        """Predict keypoint positions using linear interpolation based on history."""
        predicted_points = {}
        for kp_name in self.keypoint_names:
            history = self.track_history[kp_name]
            if len(history) >= 2:
                # Use the last two known points for prediction
                p1 = history[-2]  # (frame, x, y)
                p2 = history[-1]  # (frame, x, y)
                
                # Simple linear extrapolation/interpolation using p1 and p2 movement vector
                dx = p2[1] - p1[1] # change in X
                dy = p2[2] - p1[2] # change in Y
                
                # Predict for the current frame_idx (which is end_frame + 1)
                current_x = p2[1] + dx * 0.5 # Predict half a step forward
                current_y = p2[2] + dy * 0.5
                predicted_points[kp_name] = (int(current_x), int(current_y))
            else:
                # If not enough history, return None or last known point if available
                if self.track_history[kp_name]:
                    last_frame, x, y = self.track_history[kp_name][-1]
                    predicted_points[kp_name] = (x, y)
        return predicted_points

    def get_manual_labels(self):
        """Get the set of labels that have been manually saved"""
        manual_labels = set()
        for labeled in self.labeled_frames:
            if labeled.get('manual', False):
                manual_labels.update(labeled['points'].keys())
        return manual_labels

    def update_label_selection_from_manual(self):
        """Update the UI to only show labels that exist in manual frames"""
        manual_labels = self.get_manual_labels()
        
        # Hide or disable labels not in manual set
        for name in self.keypoint_names:
            if name not in manual_labels:
                # You could disable these buttons or hide them
                self.kp_buttons[name]['radio'].config(state='disabled')
            else:
                self.kp_buttons[name]['radio'].config(state='normal')

    def _map_coco_to_keypoints(self, keypoints, frame_shape):
        """Legacy COCO-pose mapping. Disabled: the pretrained pose model only
        detects generic human-skeleton points, not arbitrary user-defined
        keypoints, so it cannot meaningfully map to them. Use optical flow
        (Track Between Manual) instead. Returns no points."""
        return {}
    
    def _map_to_keypoint(self, class_id, cx, cy, existing_points):
        """Map YOLO class to your keypoint names - simpler fallback"""
        # This is a simplified mapping for when pose estimation fails
        available = [k for k in self.keypoint_names if k not in existing_points]
        if available:
            # Simple assignment in order
            return available[0] if available else None
        return None
    
    def track_with_manual_assistance(self):
        """Interactive tracking: User labels some frames, YOLO tracks the rest"""
        manual_frames = [f for f in self.labeled_frames if f.get('manual', False)]
        
        if len(manual_frames) < 2:
            messagebox.showwarning("Need Manual Frames", 
                "Please label at least 2 frames manually for reference")
            return False
        
        # Sort manual frames
        manual_frames.sort(key=lambda x: x['frame'])
        
        # Track between manual frames
        total_tracked = 0
        for i in range(len(manual_frames) - 1):
            start_frame = manual_frames[i]['frame']
            end_frame = manual_frames[i + 1]['frame']
            
            if end_frame - start_frame > 1:
                self.status_var.set(f"🔄 Tracking segment {i+1}/{len(manual_frames)-1}: frames {start_frame}-{end_frame}")
                self.root.update()
                
                # Set the current frame to start frame for tracking
                self.current_frame = start_frame
                success = self.track_with_yolo(
                    start_frame=start_frame + 1,
                    end_frame=end_frame - 1,
                    use_manual_anchors=False
                )
                if success:
                    total_tracked += (end_frame - start_frame - 1)
        
        self.status_var.set(f"✅ Manual-assist tracking complete! Added {total_tracked} frames")
        return True
    
    def hybrid_track(self):
        """Hybrid approach: YOLO + manual verification points"""
        if not self.cap:
            messagebox.showwarning("No Video", "Please load a video first")
            return
        
        # 1. Find all manual frames
        manual_frames = [f for f in self.labeled_frames if f.get('manual', False)]
        
        if len(manual_frames) < 2:
            messagebox.showwarning("Need Manual Frames", 
                "Please label at least 2 frames manually")
            return
        
        # 2. Remove existing tracked frames
        self.labeled_frames = [f for f in self.labeled_frames if f.get('manual', False)]
        
        # 3. Run YOLO tracking between manual frames
        manual_frames.sort(key=lambda x: x['frame'])
        
        total_tracked = 0
        for i in range(len(manual_frames) - 1):
            start = manual_frames[i]
            end = manual_frames[i + 1]
            
            if end['frame'] - start['frame'] <= 1:
                continue
            
            # Set current frame to start
            self.current_frame = start['frame']
            
            # Track segment
            self.status_var.set(f"🔄 Tracking segment: frames {start['frame']}-{end['frame']}")
            self.root.update()
            
            success = self.track_with_yolo(
                start_frame=start['frame'] + 1,
                end_frame=end['frame'] - 1,
                use_manual_anchors=False
            )
            
            if success:
                total_tracked += (end['frame'] - start['frame'] - 1)
        
        # 4. Final interpolation for any gaps
        self.smart_interpolate()
        
        self.status_var.set(f"✅ Hybrid tracking complete! Manual + YOLO + Interpolation")
        self.show_frame(self.current_frame)
    
        # === APPLY SMOOTHING AFTER ALL FRAMES ARE POPULATED ===
        smoothed_count = self.apply_moving_average() 
        self.status_var.set(f"✅ Hybrid tracking complete! Manual + YOLO + Interpolation + Smoothing (Smoothed {smoothed_count} points)")


    # ============ SPACEBAR HANDLING ============
    
    def on_space_press(self, event=None):
        """Handle spacebar press - toggle playback only once"""
        if not self.space_held:
            self.space_held = True
            self.toggle_play()
        return "break"

    def on_space_release(self, event=None):
        """Handle spacebar release - reset the held flag"""
        self.space_held = False
        return "break"
    
    # ============ PATH NORMALIZATION ============
    
    def normalize_path(self, path):
        return str(Path(path).resolve())
    
    # ============ PROGRESS TRACKING ============
    
    def load_progress(self):
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    self.progress_data = json.load(f)
                self.status_var.set(f"📊 Loaded progress: {len(self.progress_data)} videos tracked")
            except:
                self.progress_data = {}
                self.status_var.set("No valid progress file found")
        else:
            self.progress_data = {}
            self.status_var.set("No progress file found - starting fresh")
    
    def save_progress(self):
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except Exception as e:
            self.status_var.set(f"⚠️ Could not save progress: {e}")
    
    def save_frame_progress(self):
        if not self.video_path:
            return
        
        video_key = self.normalize_path(self.video_path)
        self.progress_data[video_key] = {
            'completed': False,
            'frames_labeled': len(self.labeled_frames),
            'total_frames': self.total_frames,
            'last_frame': self.current_frame,
            'last_updated': datetime.now().isoformat(),
            'video_name': os.path.basename(self.video_path),
            'labeled_frames': self.labeled_frames
        }
        self.save_progress()
        self.update_video_progress()
    
    def load_saved_progress(self, video_path):
        video_key = self.normalize_path(video_path)
        if video_key in self.progress_data:
            data = self.progress_data[video_key]
            if 'labeled_frames' in data:
                self.labeled_frames = data['labeled_frames']
                self.progress_bar['value'] = len(self.labeled_frames)
                self.progress_label.config(text=f"{len(self.labeled_frames)} / {self.total_frames} frames")
                self.status_var.set(f"🔄 Restored {len(self.labeled_frames)} labeled frames")
            return data
        return None
    
    def mark_video_complete(self, video_path):
        video_key = self.normalize_path(video_path)
        self.progress_data[video_key] = {
            'completed': True,
            'frames_labeled': len(self.labeled_frames),
            'total_frames': self.total_frames,
            'last_frame': self.current_frame,
            'completion_date': datetime.now().isoformat(),
            'video_name': os.path.basename(video_path),
            'labeled_frames': self.labeled_frames
        }
        self.save_progress()
        self.update_video_progress()

    def get_video_status(self, video_path):
        video_key = self.normalize_path(video_path)
        return self.progress_data.get(video_key)

    def update_video_progress(self):
        if self.video_files:
            total = len(self.video_files)
            completed = 0
            for v in self.video_files:
                status = self.get_video_status(v)
                if status and status.get('completed', False):
                    completed += 1
            
            self.video_progress_label.config(text=f"📊 {completed}/{total} videos done")
            self.root.title(f"🎬 Video Labeler - {completed}/{total} videos done")
        else:
            if self.video_path:
                self.video_progress_label.config(text="📊 Single video mode")
                self.root.title(f"🎬 Video Labeler - {os.path.basename(self.video_path)}")
            else:
                self.video_progress_label.config(text="📊 No videos loaded")
                self.root.title("🎬 Video Labeler - AI-Powered Tracking")
    
    # ============ UI SETUP ============
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Video display
        video_frame = ttk.LabelFrame(main_frame, text="Video Player", padding=5)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_panel = tk.Canvas(video_frame, bg='black')
        self.video_panel.pack(fill=tk.BOTH, expand=True)
        self.video_panel.bind("<Button-1>", self.on_canvas_click)
        self.video_panel.bind("<MouseWheel>", self.on_mouse_wheel)
        self.video_panel.bind("<ButtonPress-2>", self.on_pan_start)
        self.video_panel.bind("<B2-Motion>", self.on_pan_move)
        self.video_panel.bind("<Button-1>", lambda e: self.video_panel.focus_set(), add="+")
        
        # Video controls
        controls = ttk.Frame(video_frame)
        controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(controls, text="⏮", command=self.prev_frame, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="⏪", command=self.prev_10, width=3).pack(side=tk.LEFT, padx=2)
        self.play_btn = ttk.Button(controls, text="▶ Play", command=self.toggle_play, width=6)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="⏩", command=self.next_10, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="⏭", command=self.next_frame, width=3).pack(side=tk.LEFT, padx=2)
        
        self.slider = ttk.Scale(controls, from_=0, to=100, orient=tk.HORIZONTAL)
        self.slider.bind("<ButtonRelease-1>", self.slider_changed)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.frame_label = ttk.Label(controls, text="0 / 0")
        self.frame_label.pack(side=tk.RIGHT, padx=5)
        
        # Navigation controls
        nav_frame = ttk.Frame(video_frame)
        nav_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(nav_frame, text="⏮ Previous Video", 
                  command=self.prev_video, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="⏭ Next Video", 
                  command=self.next_video, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="📊 Show Progress", 
                  command=self.show_progress, width=15).pack(side=tk.LEFT, padx=2)
        
        self.video_progress_label = ttk.Label(nav_frame, text="📊 0/0 videos done", foreground='blue')
        self.video_progress_label.pack(side=tk.RIGHT, padx=5)

        # Right panel (scrollable so controls never get clipped on small windows)
        right_outer = ttk.LabelFrame(main_frame, text="Controls", padding=2)
        right_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        right_canvas = tk.Canvas(right_outer, borderwidth=0, highlightthickness=0, width=240)
        right_vsb = ttk.Scrollbar(right_outer, orient=tk.VERTICAL, command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_vsb.set)
        right_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(right_canvas, padding=8)
        self._right_window = right_canvas.create_window((0, 0), window=right_frame, anchor='nw')
        self.right_canvas = right_canvas
        self.right_frame_inner = right_frame

        right_frame.bind("<Configure>",
                         lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all")))
        right_canvas.bind("<Configure>",
                          lambda e: right_canvas.itemconfig(self._right_window, width=e.width))

        def _on_right_wheel(event):
            right_canvas.yview_scroll(int(-event.delta / 120), "units")
        right_canvas.bind("<Enter>", lambda e: right_canvas.bind_all("<MouseWheel>", _on_right_wheel))
        right_canvas.bind("<Leave>", lambda e: right_canvas.unbind_all("<MouseWheel>"))
        
        # File controls
        ttk.Label(right_frame, text="📁 File", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,5))
        ttk.Button(right_frame, text="Open Video", command=self.open_video, width=15).pack(pady=2)
        ttk.Button(right_frame, text="Open Folder", command=self.open_folder, width=15).pack(pady=2)
        
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Keypoint section (select to place, edit name inline, add/remove)
        ttk.Label(right_frame, text="🎯 Keypoints",
                  font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(right_frame, text="Click ◉ to select · edit name · ✕ to remove",
                  font=('Arial', 8), foreground='gray').pack(anchor=tk.W)

        # Shared selection variable + rebuildable container for the keypoint rows
        self.kp_select_var = tk.StringVar(value="")
        self.kp_container = ttk.Frame(right_frame)
        self.kp_container.pack(fill=tk.X)
        self.kp_buttons = {}

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Quick select shortcuts (also rebuilt when keypoints change)
        ttk.Label(right_frame, text="⌨️ Shortcuts:", font=('Arial', 9)).pack(anchor=tk.W)
        self.kp_shortcut_container = ttk.Frame(right_frame)
        self.kp_shortcut_container.pack(fill=tk.X, pady=2)

        # Populate keypoint rows + shortcut hints
        self._populate_keypoint_buttons()

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Label actions
        ttk.Label(right_frame, text="📝 Actions", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,5))
        ttk.Button(right_frame, text="✅ Save Frame", command=self.save_frame, width=15).pack(pady=2)
        ttk.Button(right_frame, text="🚫 Mark Occluded (H)", command=self.mark_occluded, width=15).pack(pady=2)
        ttk.Button(right_frame, text="↩️ Undo Last", command=self.undo_point, width=15).pack(pady=2)
        ttk.Button(right_frame, text="🗑️ Clear Frame", command=self.clear_frame, width=15).pack(pady=2)
        ttk.Button(right_frame, text="⏭️ Skip Frame", command=self.skip_frame, width=15).pack(pady=2)
        ttk.Button(right_frame, text="✅ Mark Complete", command=self.mark_current_complete, width=15).pack(pady=2)
        
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Optical Flow tracking (recommended for custom keypoints)
        ttk.Label(right_frame, text="🌊 Optical Flow (recommended)",
                  font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(right_frame,
                  text="Fills frames BETWEEN your manual\nlabels by following the real pixels.\nLands exactly on each manual frame.",
                  font=('Arial', 8), foreground='gray', justify=tk.LEFT).pack(anchor=tk.W)
        ttk.Button(right_frame, text="🌊 Track Between Manual (O)",
                   command=self.optical_flow_track, width=22).pack(pady=4)

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # YOLO Tracking controls
        ttk.Label(right_frame, text="🤖 YOLO Tracking", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,5))
        ttk.Label(right_frame,
                  text="⚠️ Pre-trained pose can't detect your\ncustom points — expect drift. Use only\nafter training your own model.",
                  font=('Arial', 8), foreground='#aa5500', justify=tk.LEFT).pack(anchor=tk.W)
        
        tracker_frame = ttk.Frame(right_frame)
        tracker_frame.pack(fill=tk.X, pady=2)
        
        # Add checkbox for manual labels only
        self.manual_labels_only = tk.BooleanVar(value=True)
        ttk.Checkbutton(tracker_frame, text="🎯 Only track manual labels", 
                        variable=self.manual_labels_only).pack(anchor=tk.W)

        self.tracker_var = tk.StringVar(value="botsort")
        ttk.Radiobutton(tracker_frame, text="BoTSORT (Balanced)", 
                       variable=self.tracker_var, value="botsort").pack(anchor=tk.W)
        ttk.Radiobutton(tracker_frame, text="ByteTrack (Fast)", 
                       variable=self.tracker_var, value="bytetrack").pack(anchor=tk.W)
        
        # YOLO confidence threshold
        conf_frame = ttk.Frame(right_frame)
        conf_frame.pack(fill=tk.X, pady=2)
        ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT)
        self.conf_scale = ttk.Scale(conf_frame, from_=0.1, to=0.9, orient=tk.HORIZONTAL, length=80)
        self.conf_scale.set(0.5)
        self.conf_scale.pack(side=tk.LEFT, padx=5)
        self.conf_label = ttk.Label(conf_frame, text="0.5")
        self.conf_label.pack(side=tk.LEFT)
        self.conf_scale.configure(command=lambda v: self.conf_label.config(text=f"{float(v):.1f}"))
        
        ttk.Button(right_frame, text="🎯 YOLO Track Current", 
                  command=self.track_with_yolo, width=15).pack(pady=2)
        ttk.Button(right_frame, text="🔄 Hybrid Track (Manual+YOLO)", 
                  command=self.hybrid_track, width=15).pack(pady=2)
        ttk.Button(right_frame, text="📊 Track All Frames", 
                  command=lambda: self.track_with_yolo(0, self.total_frames-1), width=15).pack(pady=2)
        ttk.Button(right_frame, text="🧹 Clear Tracked", 
                  command=self.clear_tracked_frames, width=15).pack(pady=2)
        
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Backup interpolation
        ttk.Label(right_frame, text="🔄 Backup Interpolation", font=('Arial', 9)).pack(anchor=tk.W)
        ttk.Button(right_frame, text="📐 Linear Interpolate", 
                  command=self.linear_interpolate_all, width=15).pack(pady=1)
        
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Adjustment controls
        ttk.Label(right_frame, text="🎯 Adjust Points", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,5))
        
        adjust_frame = ttk.Frame(right_frame)
        adjust_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(adjust_frame, text="◀", command=lambda: self.adjust_point(-5, 0), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(adjust_frame, text="▲", command=lambda: self.adjust_point(0, -5), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(adjust_frame, text="▼", command=lambda: self.adjust_point(0, 5), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(adjust_frame, text="▶", command=lambda: self.adjust_point(5, 0), width=3).pack(side=tk.LEFT, padx=1)
        
        adjust_frame2 = ttk.Frame(right_frame)
        adjust_frame2.pack(fill=tk.X, pady=2)
        
        ttk.Button(adjust_frame2, text="Fine ◀", command=lambda: self.adjust_point(-1, 0), width=5).pack(side=tk.LEFT, padx=1)
        ttk.Button(adjust_frame2, text="Fine ▲", command=lambda: self.adjust_point(0, -1), width=5).pack(side=tk.LEFT, padx=1)
        ttk.Button(adjust_frame2, text="Fine ▼", command=lambda: self.adjust_point(0, 1), width=5).pack(side=tk.LEFT, padx=1)
        ttk.Button(adjust_frame2, text="Fine ▶", command=lambda: self.adjust_point(1, 0), width=5).pack(side=tk.LEFT, padx=1)
        
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Progress
        ttk.Label(right_frame, text="📊 Progress", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,5))
        self.progress_bar = ttk.Progressbar(right_frame, length=150, mode='determinate')
        self.progress_bar.pack(pady=5)
        self.progress_label = ttk.Label(right_frame, text="0 / 0 frames")
        self.progress_label.pack()
        
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Export
        ttk.Label(right_frame, text="💾 Export", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,5))
        ttk.Button(right_frame, text="Export Labels", command=self.export_labels, width=15).pack(pady=2)
        
        # Status bar
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        # Select first keypoint
        if self.keypoint_names:
            self.select_keypoint(self.keypoint_names[0])
        
        self.video_panel.focus_set()

    def _finalize_window(self):
        """Force a clean initial layout so the window opens at the right size
        instead of needing a manual resize."""
        self.root.update_idletasks()
        # Size the scrollable control panel to its natural width so nothing clips
        try:
            req_w = self.right_frame_inner.winfo_reqwidth()
            self.right_canvas.configure(width=req_w)
        except Exception:
            pass
        self.root.update_idletasks()
        # Fit the window to its content, clamped to the screen, and centered
        req_w = self.root.winfo_reqwidth()
        req_h = self.root.winfo_reqheight()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(max(req_w, 1100), screen_w - 80)
        win_h = min(max(req_h, 750), screen_h - 80)
        x = max((screen_w - win_w) // 2, 0)
        y = max((screen_h - win_h) // 3, 0)
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.minsize(1000, 650)

    def bind_shortcuts(self):
        """Bind keyboard shortcuts globally"""
        self.root.bind_all('<KeyPress-space>', self.on_space_press)
        self.root.bind_all('<KeyRelease-space>', self.on_space_release)
        self.root.bind_all('<Right>', self.next_frame)
        self.root.bind_all('<Left>', self.prev_frame)
        self.root.bind_all('<Control-s>', self.save_frame)
        self.root.bind_all('<Control-z>', self.undo_point)
        self.root.bind_all('<Escape>', self.skip_frame)
        self.root.bind_all('<h>', self.mark_occluded)
        self.root.bind_all('<H>', self.mark_occluded)
        for i in range(9):  # support up to 9 keypoints via number keys
            self.root.bind_all(f'<Key-{i+1}>', lambda e, idx=i: self.select_keypoint_by_index(idx))
        self.root.bind_all('<t>', lambda e: self.track_with_yolo())
        self.root.bind_all('<o>', lambda e: self.optical_flow_track())
    
    # ============ KEYPOINT SELECTION ============
    
    def select_keypoint(self, name):
        if name not in self.keypoint_names:
            return
            
        self.current_kp = self.keypoint_names.index(name)
        self.kp_select_var.set(name)

        for kp_name, data in self.kp_buttons.items():
            if kp_name == name:
                data['color'].configure(bg='yellow')
                self.status_var.set(f"Selected: {self.keypoint_display_names[name]} - Click on video to place")
            else:
                if data['placed']:
                    data['color'].configure(bg='green')
                else:
                    data['color'].configure(bg='gray')
        
        if self.cap:
            self.slider_update = False
            self.show_frame(self.current_frame, update_slider=False)
            self.slider_update = True
    
    def select_keypoint_by_index(self, idx, event=None):
        if idx < len(self.keypoint_names):
            self.select_keypoint(self.keypoint_names[idx])
        return "break"
    
    # ============ VIDEO LOADING ============
    
    def open_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.video_files = []
            self.current_video_idx = 0
            self.load_video(file_path)
            self.status_var.set(f"📹 Single video mode: {os.path.basename(file_path)}")
    
    def open_folder(self):
        folder = filedialog.askdirectory(title="Select folder with videos")
        if folder:
            videos = set()
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI', '*.MOV', '*.MKV']:
                videos.update(Path(folder).glob(ext))
            
            if videos:
                self.video_files = sorted(videos, key=natural_sort_key)
                self.current_video_idx = 0
                self.status_var.set(f"📁 Found {len(videos)} videos in folder")
                self.update_video_progress()
                self.find_first_uncompleted()
            else:
                messagebox.showwarning("No Videos", "No video files found in this folder")
                self.video_files = []
                self.current_video_idx = 0
                self.update_video_progress()
    
    def find_first_uncompleted(self):
        if not self.video_files:
            return
        
        for i, video_path in enumerate(self.video_files):
            status = self.get_video_status(video_path)
            if not status or not status.get('completed', False):
                self.current_video_idx = i
                self.load_video(str(video_path))
                self.status_var.set(f"📹 Starting with: {os.path.basename(video_path)}")
                return
        
        self.current_video_idx = 0
        self.load_video(str(self.video_files[0]))
        messagebox.showinfo("All Done!", "All videos completed!\nLoading first video anyway.")
        self.status_var.set("🎉 All videos completed!")
    
    def load_video(self, path):
        self.video_path = path
        self.current_video_exported = False
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open video:\n{path}")
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.current_frame = 0
        self.points = {}
        self.occluded = set()
        self.labeled_frames = []
        self.tracking_data = {}
        self.tracking_active = False
        
        for name in self.keypoint_names:
            self.kp_buttons[name]['placed'] = False
            self.kp_buttons[name]['color'].configure(bg='gray')
            self.kp_buttons[name]['status'].configure(text='⬜', foreground='gray')
        
        if self.keypoint_names:
            self.select_keypoint(self.keypoint_names[0])
        
        self.slider.config(to=self.total_frames-1)
        self.frame_label.config(text=f"0 / {self.total_frames}")
        self.progress_bar['maximum'] = self.total_frames
        
        saved_data = self.load_saved_progress(path)
        
        if saved_data:
            if saved_data.get('completed', False):
                self.status_var.set(f"✅ Already completed: {os.path.basename(path)}")
                self.progress_bar['value'] = self.total_frames
                self.progress_label.config(text=f"{self.total_frames} / {self.total_frames} frames")
                self.show_frame(0)
            else:
                last_frame = saved_data.get('last_frame', 0)
                if last_frame > 0 and last_frame < self.total_frames:
                    self.current_frame = last_frame
                    self.status_var.set(f"🔄 Resuming from frame {last_frame}")
                    self.show_frame(last_frame)
                else:
                    self.show_frame(0)
        else:
            self.show_frame(0)
            self.progress_bar['value'] = 0
            self.progress_label.config(text=f"0 / {self.total_frames} frames")
        
        self.status_var.set(f"📹 Loaded: {os.path.basename(path)} ({self.total_frames} frames)")
        self.update_video_progress()
        
        if self.video_files:
            idx = self.current_video_idx + 1
            total = len(self.video_files)
            self.status_var.set(f"📹 Video {idx}/{total}: {os.path.basename(path)} ({self.total_frames} frames)")
        
        self.video_panel.focus_set()
        self.show_frame(0, update_slider=True)
    
    # ============ VIDEO DISPLAY ============
    
    def on_canvas_click(self, event):
        if not self.cap:
            return

        if not self.keypoint_names or self.current_kp >= len(self.keypoint_names):
            self.status_var.set("⚠️ No keypoints defined. Use ➕ Add Keypoint in the 🎯 Keypoints panel first.")
            return

        name = self.keypoint_names[self.current_kp]
        x, y = event.x, event.y

        self.points[name] = (x, y)
        self.occluded.discard(name)  # placing a position clears any occluded mark
        self.kp_buttons[name]['placed'] = True
        self.kp_buttons[name]['color'].configure(bg='green')
        self.kp_buttons[name]['status'].configure(text='✅', foreground='green')
        
        self.status_var.set(f"✅ Placed: {self.keypoint_display_names[name]} at ({x}, {y})")
        
        self.advance_to_next_keypoint()
        
        self.slider_update = False
        self.show_frame(self.current_frame)
        self.slider_update = True
    
    def advance_to_next_keypoint(self):
        # A keypoint is "done" if placed OR explicitly marked occluded
        def done(name):
            return self.kp_buttons[name]['placed'] or name in self.occluded

        if all(done(name) for name in self.keypoint_names):
            self.status_var.set("All keypoints placed/occluded! Press 'Save Frame' or Ctrl+S")
            return

        for name in self.keypoint_names:
            if not done(name):
                self.select_keypoint(name)
                break

    def mark_occluded(self, event=None):
        """Mark the selected keypoint as hidden/inside for this frame (visibility 0)
        instead of placing a position. Interpolation/optical-flow skip it, so no
        bounding box is invented while the part is occluded."""
        if not self.keypoint_names or self.current_kp >= len(self.keypoint_names):
            return "break"
        name = self.keypoint_names[self.current_kp]
        self.points.pop(name, None)          # remove any placed position
        self.occluded.add(name)
        self.kp_buttons[name]['placed'] = False
        self.kp_buttons[name]['color'].configure(bg='#663333')
        self.kp_buttons[name]['status'].configure(text='🚫', foreground='#cc7777')
        self.status_var.set(f"🚫 {name}: occluded / inside (no box this frame)")
        self.advance_to_next_keypoint()
        self.slider_update = False
        self.show_frame(self.current_frame)
        self.slider_update = True
        return "break"
    
    def show_frame(self, frame_idx, update_slider=True):
        if not self.cap:
            return
        
        if frame_idx < 0:
            frame_idx = 0
        elif frame_idx >= self.total_frames:
            frame_idx = self.total_frames - 1
        
        self.current_frame = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        
        h, w = frame.shape[:2]
        display_h = 600
        scale = display_h / h
        display_w = int(w * scale)
        frame = cv2.resize(frame, (display_w, display_h))
        
        # Draw current points being placed
        for name, coords in self.points.items():
            color = self.colors[name]
            cv2.circle(frame, coords, 14, color, 2)
            cv2.circle(frame, coords, 10, color, -1)
            cv2.circle(frame, coords, 12, (255, 255, 255), 1)
            cv2.putText(frame, name.replace('_', ' ').title(), 
                    (coords[0]+15, coords[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw saved frames (manual + tracked)
        for labeled in self.labeled_frames:
            if labeled['frame'] == frame_idx:
                for name, coords in labeled['points'].items():
                    color = self.colors[name]
                    is_tracked = labeled.get('tracked', False)
                    is_manual = labeled.get('manual', False)
                    
                    if is_tracked:
                        # Draw tracked points with different style
                        cv2.circle(frame, coords, 12, color, 2)
                        cv2.circle(frame, coords, 6, color, -1)
                        cv2.putText(frame, f"{name} (YOLO)", 
                                  (coords[0]+15, coords[1]+15), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                    else:
                        # Manual points
                        cv2.circle(frame, coords, 14, color, 2)
                        cv2.circle(frame, coords, 10, color, -1)
                        cv2.circle(frame, coords, 12, (255, 255, 255), 1)
                        cv2.putText(frame, name.replace('_', ' ').title(), 
                                  (coords[0]+15, coords[1]-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show currently selected keypoint (skip the prompt if it's occluded)
        if self.current_kp < len(self.keypoint_names):
            name = self.keypoint_names[self.current_kp]
            if not self.kp_buttons[name]['placed'] and name not in self.occluded:
                cv2.line(frame, (display_w//2 - 30, display_h//2), 
                        (display_w//2 + 30, display_h//2), self.colors[name], 2)
                cv2.line(frame, (display_w//2, display_h//2 - 30), 
                        (display_w//2, display_h//2 + 30), self.colors[name], 2)
                cv2.putText(frame, f"Click to place: {self.keypoint_display_names[name]}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors[name], 2)
        
        # Show frame info
        manual_count = sum(1 for f in self.labeled_frames if f.get('manual', False))
        tracked_count = sum(1 for f in self.labeled_frames if f.get('tracked', False))
        cv2.putText(frame, f"Frame: {frame_idx}/{self.total_frames} | Manual: {manual_count} Tracked: {tracked_count}", 
                (display_w-450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show tracker status
        if self.tracking_active:
            cv2.putText(frame, "🔴 YOLO TRACKING ACTIVE", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Convert to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_panel.imgtk = imgtk
        self.video_panel.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.video_panel.config(width=display_w, height=display_h)
        
        if update_slider:
            self.slider.set(frame_idx)
        self.frame_label.config(text=f"{frame_idx} / {self.total_frames}")
    
    def on_mouse_wheel(self, event):
        pass
    
    def on_pan_start(self, event):
        self.pan_x = event.x
        self.pan_y = event.y
    
    def on_pan_move(self, event):
        pass
    
    # ============ VIDEO CONTROLS ============
    
    def toggle_play(self, event=None):
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="▶ Play")
            if hasattr(self, '_after_id'):
                try:
                    self.root.after_cancel(self._after_id)
                except:
                    pass
        else:
            self.is_playing = True
            self.play_btn.config(text="⏸ Pause")
            self.play_video()
        return "break"
    
    def play_video(self):
        if not self.is_playing or not self.cap:
            return
        
        next_frame = self.current_frame + 1
        if next_frame >= self.total_frames:
            self.is_playing = False
            self.play_btn.config(text="▶ Play")
            return
        
        self.show_frame(next_frame, update_slider=True)
        self._after_id = self.root.after(1000 // self.fps, self.play_video)

    def next_frame(self, event=None):
        if not self.cap:
            return "break"
        
        if self.current_frame < self.total_frames - 1:
            self.show_frame(self.current_frame + 1, update_slider=True)
        else:
            if self.is_playing:
                self.stop_playback()
                self.status_var.set("🎬 End of video reached")
        return "break"
    
    def stop_playback(self):
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="▶ Play")
            if hasattr(self, '_after_id'):
                try:
                    self.root.after_cancel(self._after_id)
                except:
                    pass

    def prev_frame(self, event=None):
        if self.cap and self.current_frame > 0:
            self.show_frame(self.current_frame - 1, update_slider=True)
        return "break"

    def next_10(self):
        if self.cap and self.current_frame < self.total_frames - 10:
            self.show_frame(self.current_frame + 10, update_slider=True)

    def prev_10(self):
        if self.cap and self.current_frame > 10:
            self.show_frame(self.current_frame - 10, update_slider=True)

    def slider_changed(self, event):
        if not self.cap:
            return
        
        try:
            frame = int(float(self.slider.get()))
            if frame != self.current_frame:
                self.show_frame(frame, update_slider=False)
        except:
            pass
    
    # ============ LABELING ACTIONS ============
    
    def save_frame(self, event=None):
        if not self.cap:
            return "break"
        
        if not self.points and not self.occluded:
            # Check if there are tracked points for this frame
            existing_frame = None
            for labeled in self.labeled_frames:
                if labeled['frame'] == self.current_frame:
                    existing_frame = labeled
                    break

            if existing_frame:
                self.status_var.set(f"ℹ️ Frame {self.current_frame} already has {len(existing_frame['points'])} points")
                return "break"
            else:
                messagebox.showwarning("No Points", "Place a keypoint or mark one occluded before saving.")
                return "break"

        # Save the frame with points (+ any occluded keypoints, exported as visibility 0)
        frame_data = {
            'frame': self.current_frame,
            'timestamp': self.current_frame / self.fps,
            'points': self.points.copy(),
            'occluded': sorted(self.occluded),
            'manual': True,
            'tracked': False
        }
        
        # Check if frame already exists, replace if so
        existing_idx = None
        for i, labeled in enumerate(self.labeled_frames):
            if labeled['frame'] == self.current_frame:
                existing_idx = i
                break
        
        if existing_idx is not None:
            self.labeled_frames[existing_idx] = frame_data
            self.status_var.set(f"🔄 Updated frame {self.current_frame}")
        else:
            self.labeled_frames.append(frame_data)
            self.status_var.set(f"✅ Frame {self.current_frame} saved!")
        
        self.current_video_exported = False
        self.progress_bar['value'] = len(self.labeled_frames)
        self.progress_label.config(text=f"{len(self.labeled_frames)} / {self.total_frames} frames")
        
        if self.video_path:
            self.save_frame_progress()
        
        # Clear points
        self.points = {}
        self.occluded = set()
        for name in self.keypoint_names:
            self.kp_buttons[name]['placed'] = False
            self.kp_buttons[name]['color'].configure(bg='gray')
            self.kp_buttons[name]['status'].configure(text='⬜', foreground='gray')

        if self.keypoint_names:
            self.select_keypoint(self.keypoint_names[0])

        if self.is_playing:
            self.next_frame()
        else:
            self.slider_update = False
            self.show_frame(self.current_frame, update_slider=True)
            self.slider_update = True
        
        self.video_panel.focus_set()
        return "break"
    
    def undo_point(self, event=None):
        if self.points:
            last = list(self.points.keys())[-1]
            del self.points[last]
            self.kp_buttons[last]['placed'] = False
            self.kp_buttons[last]['color'].configure(bg='gray')
            self.kp_buttons[last]['status'].configure(text='⬜', foreground='gray')
            self.select_keypoint(last)
            self.status_var.set(f"↩️ Undo: removed {last}")
            self.slider_update = False
            self.show_frame(self.current_frame)
            self.slider_update = True
        return "break"

    def clear_frame(self):
        if self.labeled_frames:
            if not messagebox.askyesno("Clear All Frames", 
                                    f"Remove ALL {len(self.labeled_frames)} labeled frames?"):
                return
        
        self.points = {}
        self.occluded = set()
        self.labeled_frames = []
        for name in self.keypoint_names:
            self.kp_buttons[name]['placed'] = False
            self.kp_buttons[name]['color'].configure(bg='gray')
            self.kp_buttons[name]['status'].configure(text='⬜', foreground='gray')

        self.clear_saved_progress()
        
        if self.keypoint_names:
            self.select_keypoint(self.keypoint_names[0])
        
        self.status_var.set("🗑️ Cleared all labels")
        self.current_video_exported = False
        self.progress_bar['value'] = 0
        self.progress_label.config(text=f"0 / {self.total_frames} frames")
        
        self.slider_update = False
        self.show_frame(self.current_frame)
        self.slider_update = True
    
    def clear_saved_progress(self):
        if not self.video_path:
            return
        
        video_key = self.normalize_path(self.video_path)
        if video_key in self.progress_data:
            del self.progress_data[video_key]
            self.save_progress()
            self.status_var.set(f"🗑️ Cleared progress for {os.path.basename(self.video_path)}")
            self.update_video_progress()

    def skip_frame(self, event=None):
        self.points = {}
        self.occluded = set()
        for name in self.keypoint_names:
            self.kp_buttons[name]['placed'] = False
            self.kp_buttons[name]['color'].configure(bg='gray')
            self.kp_buttons[name]['status'].configure(text='⬜', foreground='gray')
        
        if self.keypoint_names:
            self.select_keypoint(self.keypoint_names[0])
        
        self.status_var.set(f"⏭️ Skipped frame {self.current_frame}")
        
        if self.is_playing:
            self.next_frame()
        else:
            self.slider_update = False
            self.show_frame(self.current_frame, update_slider=True)
            self.slider_update = True
        
        return "break"

    def mark_current_complete(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "No video loaded")
            return
        
        if not self.labeled_frames:
            if not messagebox.askyesno("No Labels", "Mark as complete anyway?"):
                return
        
        if self.labeled_frames and not self.current_video_exported:
            if messagebox.askyesno("Export First?", "Export labels before marking as complete?"):
                exported = self.export_labels()
                if not exported:
                    return
        
        self.mark_video_complete(self.video_path)
        self.status_var.set(f"✅ Marked {os.path.basename(self.video_path)} as complete")
        self.stop_playback()
        
        if self.video_files and len(self.video_files) > 1:
            if messagebox.askyesno("Next Video", "Move to next video?"):
                self._advance_to_next_video_force()
    
    # ============ INTERPOLATION METHODS ============
    
    def smart_interpolate(self):
        """Hybrid interpolation using manual frames + tracked data"""
        if len(self.labeled_frames) < 2:
            messagebox.showwarning("Need at least 2 labeled frames")
            return
        
        # Sort by frame number
        self.labeled_frames.sort(key=lambda x: x['frame'])
        
        # Remove old interpolated frames
        self.labeled_frames = [f for f in self.labeled_frames if not f.get('interpolated', False)]
        
        # Get manual frames as anchors
        anchors = [f for f in self.labeled_frames if f.get('manual', False)]
        
        if len(anchors) < 2:
            messagebox.showwarning("Need at least 2 manual frames for interpolation")
            return
        
        new_frames = []
        
        for i in range(len(anchors) - 1):
            start = anchors[i]
            end = anchors[i + 1]
            
            # Distance between frames
            distance = end['frame'] - start['frame']
            
            if distance <= 1:
                continue
            
            # Interpolate between them
            for frame_num in range(start['frame'] + 1, end['frame']):
                t = (frame_num - start['frame']) / distance
                points = {}
                
                for kp_name in self.keypoint_names:
                    if kp_name in start['points'] and kp_name in end['points']:
                        p1 = start['points'][kp_name]
                        p2 = end['points'][kp_name]
                        points[kp_name] = (
                            int(p1[0] + t * (p2[0] - p1[0])),
                            int(p1[1] + t * (p2[1] - p1[1]))
                        )
                    elif kp_name in start['points']:
                        points[kp_name] = start['points'][kp_name]
                    elif kp_name in end['points']:
                        points[kp_name] = end['points'][kp_name]
                
                if points:
                    new_frames.append({
                        'frame': frame_num,
                        'timestamp': frame_num / self.fps,
                        'points': points,
                        'manual': False,
                        'tracked': False,
                        'interpolated': True
                    })
        
        # Add interpolated frames
        self.labeled_frames.extend(new_frames)
        self.labeled_frames.sort(key=lambda x: x['frame'])
        
        # Update UI
        self.progress_bar['value'] = len(self.labeled_frames)
        self.progress_label.config(text=f"{len(self.labeled_frames)} / {self.total_frames} frames")
        self.status_var.set(f"✅ Interpolated {len(new_frames)} frames")
    
    def linear_interpolate_all(self):
        """Backup: Linear interpolation between manual frames"""
        manual_frames = [f for f in self.labeled_frames if f.get('manual', False)]
        
        if len(manual_frames) < 2:
            messagebox.showwarning("Not Enough Labels", "Need at least 2 manual frames")
            return False
        
        # Remove existing interpolated frames
        self.labeled_frames = [f for f in self.labeled_frames if f.get('manual', False)]
        
        sorted_labels = sorted(manual_frames, key=lambda x: x['frame'])
        frames_added = 0
        
        for i in range(len(sorted_labels) - 1):
            start = sorted_labels[i]
            end = sorted_labels[i + 1]
            
            if end['frame'] - start['frame'] <= 1:
                continue
            
            # Don't interpolate keypoints marked occluded in either anchor
            skip = set(start.get('occluded', [])) | set(end.get('occluded', []))

            for frame in range(start['frame'] + 1, end['frame']):
                t = (frame - start['frame']) / (end['frame'] - start['frame'])
                interpolated_points = {}

                for kp_name in self.keypoint_names:
                    if kp_name in skip:
                        continue
                    if kp_name in start['points'] and kp_name in end['points']:
                        p1 = start['points'][kp_name]
                        p2 = end['points'][kp_name]
                        interpolated_points[kp_name] = (
                            int(p1[0] + t * (p2[0] - p1[0])),
                            int(p1[1] + t * (p2[1] - p1[1]))
                        )
                    elif kp_name in start['points']:
                        interpolated_points[kp_name] = start['points'][kp_name]
                    elif kp_name in end['points']:
                        interpolated_points[kp_name] = end['points'][kp_name]
                
                if interpolated_points:
                    self.labeled_frames.append({
                        'frame': frame,
                        'timestamp': frame / self.fps,
                        'points': interpolated_points,
                        'manual': False,
                        'tracked': False,
                        'interpolated': True
                    })
                    frames_added += 1
        
        self.labeled_frames = sorted(self.labeled_frames, key=lambda x: x['frame'])
        self.progress_bar['value'] = len(self.labeled_frames)
        self.progress_label.config(text=f"{len(self.labeled_frames)} / {self.total_frames} frames")
        
        if self.video_path:
            self.save_frame_progress()
        
        self.status_var.set(f"✅ Interpolated {frames_added} frames")
        return True
    
    # ============ OPTICAL FLOW TRACKING (recommended for custom keypoints) ============

    def _get_video_dims(self):
        """Return native (width, height) of the loaded video in pixels."""
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def optical_flow_track(self):
        """Propagate manual labels to the frames in between using bidirectional
        Lucas-Kanade optical flow, blended between each pair of manual anchors.

        Unlike pre-trained YOLO pose, this follows the actual pixels you clicked,
        so it works for arbitrary custom keypoints and lands exactly on both
        manual anchors (no accumulated drift between them)."""
        if not self.cap:
            messagebox.showwarning("No Video", "Please load a video first")
            return False

        manual = sorted([f for f in self.labeled_frames if f.get('manual', False)],
                        key=lambda x: x['frame'])
        if len(manual) < 2:
            messagebox.showwarning("Need Manual Frames",
                "Label at least 2 frames manually first.\n"
                "Optical flow fills in the frames *between* your manual labels.")
            return False

        vid_w, vid_h = self._get_video_dims()
        # Labels are stored in DISPLAY space (video is resized to 600px tall in
        # show_frame). Optical flow must run in native video pixels, so convert
        # in/out using this scale: display = video * scale.
        disp_h = 600
        scale = disp_h / vid_h if vid_h else 1.0

        # Drop previously generated (non-manual) frames so re-running is clean
        self.labeled_frames = [f for f in self.labeled_frames
                               if f.get('manual', False) or f.get('manual_correction', False)]

        MAX_SEGMENT = 400  # beyond this, optical flow gets unreliable -> linear
        new_frames = []
        total_segments = len(manual) - 1
        self.tracking_active = True

        for si in range(total_segments):
            if not self.tracking_active:
                break
            a, b = manual[si], manual[si + 1]
            fa, fb = a['frame'], b['frame']
            if fb - fa <= 1:
                continue

            self.status_var.set(f"🌊 Optical flow: segment {si+1}/{total_segments} "
                                f"(frames {fa}-{fb})")
            self.root.update()

            # Fully skip only keypoints occluded in BOTH anchors. If occluded at
            # just one end, it's still tracked from the visible end and dropped on
            # loss — so the visible approach keeps a box and it vanishes when gone.
            skip = set(a.get('occluded', [])) & set(b.get('occluded', []))

            if fb - fa > MAX_SEGMENT:
                seg = self._linear_segment(fa, fb, a['points'], b['points'], skip)
                method = 'linear_far'
            else:
                seg = self._optical_flow_segment(fa, fb, a['points'], b['points'],
                                                 scale, vid_w, vid_h, skip)
                method = 'optical_flow'

            for fi, pts in seg.items():
                new_frames.append({
                    'frame': fi,
                    'timestamp': fi / self.fps,
                    'points': pts,
                    'manual': False,
                    'tracked': True,
                    'interpolated': True,
                    'tracking_method': method,
                })

        self.tracking_active = False
        self.labeled_frames.extend(new_frames)
        self.labeled_frames.sort(key=lambda x: x['frame'])

        self.progress_bar['value'] = len(self.labeled_frames)
        self.progress_label.config(text=f"{len(self.labeled_frames)} / {self.total_frames} frames")
        if self.video_path:
            self.save_frame_progress()

        self.status_var.set(f"✅ Optical flow complete: filled {len(new_frames)} frames "
                            f"between {len(manual)} manual anchors")
        self.slider_update = False
        self.show_frame(self.current_frame)
        self.slider_update = True
        return True

    def _linear_segment(self, fa, fb, A, B, skip=None):
        """Plain linear interpolation between two anchors (display-space points).
        Keypoints in `skip` (occluded in an anchor) are omitted -> visibility 0."""
        skip = skip or set()
        out = {}
        span = fb - fa
        for fi in range(fa + 1, fb):
            t = (fi - fa) / span
            pts = {}
            for k in self.keypoint_names:
                if k in skip:
                    continue
                # Only interpolate a point present at BOTH anchors. If it's at just
                # one end it's entering/leaving -> drop it (label disappears) rather
                # than carrying a stale position across the gap.
                if k in A and k in B:
                    pts[k] = (int(round(A[k][0] + t * (B[k][0] - A[k][0]))),
                              int(round(A[k][1] + t * (B[k][1] - A[k][1]))))
            if pts:
                out[fi] = pts
        return out

    def _optical_flow_segment(self, fa, fb, A, B, scale, vid_w, vid_h, skip=None):
        """Bidirectional LK optical flow between anchors A (frame fa) and B (frame fb).
        Points come in/out in DISPLAY space; tracking runs in native video pixels.
        Keypoints in `skip` (occluded in an anchor) are omitted -> visibility 0.
        Returns {frame_idx: {kp: (x, y)}} for the intermediate frames."""
        skip = skip or set()
        # Buffer grayscale frames for the whole segment (sequential read = fast)
        grays = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fa)
        for _ in range(fa, fb + 1):
            ret, fr = self.cap.read()
            if not ret:
                break
            grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        if len(grays) < (fb - fa + 1):
            return self._linear_segment(fa, fb, A, B, skip)  # couldn't read full segment

        n = fb - fa  # number of steps
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        def to_video(pt):
            return [pt[0] / scale, pt[1] / scale]

        # Forward/backward must agree within this many native pixels, else the
        # point is treated as occluded/drifted and dropped.
        fb_thresh = max(20.0, 0.02 * max(vid_w, vid_h))

        kps_f = [k for k in self.keypoint_names if k in A and k not in skip]
        kps_b = [k for k in self.keypoint_names if k in B and k not in skip]

        # --- Forward track from anchor A ---
        fwd, fwd_ok = {}, {}
        if kps_f:
            p = np.array([to_video(A[k]) for k in kps_f], dtype=np.float32).reshape(-1, 1, 2)
            ok = [True] * len(kps_f)
            for i in range(1, n + 1):
                p2, st, err = cv2.calcOpticalFlowPyrLK(grays[i - 1], grays[i], p, None, **lk)
                fp, fo = {}, {}
                for j, k in enumerate(kps_f):
                    x, y = float(p2[j, 0, 0]), float(p2[j, 0, 1])
                    good = (ok[j] and st[j, 0] == 1 and err[j, 0] < 30.0
                            and 0 <= x < vid_w and 0 <= y < vid_h)
                    ok[j] = good
                    fp[k], fo[k] = (x, y), good
                fwd[fa + i], fwd_ok[fa + i] = fp, fo
                p = p2

        # --- Backward track from anchor B ---
        bwd, bwd_ok = {}, {}
        if kps_b:
            p = np.array([to_video(B[k]) for k in kps_b], dtype=np.float32).reshape(-1, 1, 2)
            ok = [True] * len(kps_b)
            for i in range(1, n + 1):
                idx = n - i
                p2, st, err = cv2.calcOpticalFlowPyrLK(grays[idx + 1], grays[idx], p, None, **lk)
                bp, bo = {}, {}
                for j, k in enumerate(kps_b):
                    x, y = float(p2[j, 0, 0]), float(p2[j, 0, 1])
                    good = (ok[j] and st[j, 0] == 1 and err[j, 0] < 30.0
                            and 0 <= x < vid_w and 0 <= y < vid_h)
                    ok[j] = good
                    bp[k], bo[k] = (x, y), good
                bwd[fa + idx], bwd_ok[fa + idx] = bp, bo
                p = p2

        # --- Blend forward + backward, weighted by position between anchors ---
        out = {}
        for fi in range(fa + 1, fb):
            t = (fi - fa) / float(n)
            pts = {}
            for k in self.keypoint_names:
                if k in skip:
                    continue
                f_ok = fwd_ok.get(fi, {}).get(k, False)
                b_ok = bwd_ok.get(fi, {}).get(k, False)
                if f_ok and b_ok:
                    fx, fy = fwd[fi][k]; bx, by = bwd[fi][k]
                    # If forward (from A) and backward (from B) disagree a lot, the
                    # point is unreliable — each drifted into an occlusion — so drop
                    # it (the label disappears) instead of averaging to a wrong spot.
                    if (fx - bx) ** 2 + (fy - by) ** 2 > fb_thresh * fb_thresh:
                        continue
                    vx, vy = (1 - t) * fx + t * bx, (1 - t) * fy + t * by
                elif f_ok:
                    vx, vy = fwd[fi][k]
                elif b_ok:
                    vx, vy = bwd[fi][k]
                else:
                    # Tracker lost this point on both sides -> it left / went
                    # occluded -> drop it so the label disappears.
                    continue
                pts[k] = (int(round(vx * scale)), int(round(vy * scale)))  # back to display
            if pts:
                out[fi] = pts
        return out

    def clear_tracked_frames(self):
        """Remove all tracked frames from labeled_frames"""
        tracked_count = sum(1 for f in self.labeled_frames if f.get('tracked', False))
        
        if tracked_count == 0:
            messagebox.showinfo("No Tracked Frames", "No tracked frames to clear")
            return
        
        if messagebox.askyesno("Clear Tracked", f"Remove all {tracked_count} tracked frames?"):
            self.labeled_frames = [f for f in self.labeled_frames if not f.get('tracked', False)]
            
            self.progress_bar['value'] = len(self.labeled_frames)
            self.progress_label.config(text=f"{len(self.labeled_frames)} / {self.total_frames} frames")
            
            if self.video_path:
                self.save_frame_progress()
            
            self.status_var.set(f"🧹 Removed {tracked_count} tracked frames")
            
            self.slider_update = False
            self.show_frame(self.current_frame)
            self.slider_update = True
    
    def adjust_point(self, dx, dy):
        """Adjust the selected point by dx, dy pixels"""
        if not self.points:
            # Check if current frame has points
            current_frame_points = None
            for labeled in self.labeled_frames:
                if labeled['frame'] == self.current_frame:
                    current_frame_points = labeled['points']
                    break
            
            if current_frame_points:
                # Adjust the current keypoint in the frame
                name = self.keypoint_names[self.current_kp]
                if name in current_frame_points:
                    x, y = current_frame_points[name]
                    new_point = (max(0, x + dx), max(0, y + dy))
                    
                    # Update in labeled_frames
                    for labeled in self.labeled_frames:
                        if labeled['frame'] == self.current_frame:
                            labeled['points'][name] = new_point
                            break
                    
                    self.status_var.set(f"🔧 Adjusted {name} to {new_point}")
                    
                    # Update display
                    self.slider_update = False
                    self.show_frame(self.current_frame)
                    self.slider_update = True
                    
                    # Mark as not exported
                    self.current_video_exported = False
                    
                    if self.video_path:
                        self.save_frame_progress()
                else:
                    self.status_var.set(f"⚠️ {name} not found on this frame")
            else:
                self.status_var.set("⚠️ No points on current frame")
        else:
            # Adjust points being placed
            name = self.keypoint_names[self.current_kp]
            if name in self.points:
                x, y = self.points[name]
                self.points[name] = (max(0, x + dx), max(0, y + dy))
                self.status_var.set(f"🔧 Adjusted {name} to {self.points[name]}")
                
                self.slider_update = False
                self.show_frame(self.current_frame)
                self.slider_update = True
    
    # ============ NAVIGATION ============
    
    def _advance_to_next_video_force(self):
        if not self.video_files:
            return
        
        self.stop_playback()
        
        for i in range(self.current_video_idx + 1, len(self.video_files)):
            video_path = self.video_files[i]
            status = self.get_video_status(video_path)
            if not status or not status.get('completed', False):
                self.current_video_idx = i
                self.load_video(str(video_path))
                self.status_var.set(f"⏭ Moved to: {os.path.basename(video_path)}")
                return
        
        for i in range(0, self.current_video_idx + 1):
            video_path = self.video_files[i]
            status = self.get_video_status(video_path)
            if not status or not status.get('completed', False):
                self.current_video_idx = i
                self.load_video(str(video_path))
                self.status_var.set(f"⏭ Looped to: {os.path.basename(video_path)}")
                return
        
        self.current_video_idx = 0
        self.load_video(str(self.video_files[0]))
        messagebox.showinfo("🎉 All Done!", "All videos completed!")
        self.status_var.set("🎉 All videos completed!")
    
    def next_video(self):
        if not self.video_files:
            messagebox.showinfo("No Videos", "Open a folder first")
            return
        
        if self.labeled_frames and not self.current_video_exported:
            status = self.get_video_status(self.video_path)
            if status and status.get('completed', False):
                self._advance_to_next_video_force()
                return
                
            if not messagebox.askyesno("Export Labels First", 
                                    f"You have {len(self.labeled_frames)} labeled frames.\nExport before moving?"):
                self._advance_to_next_video_force()
                return
            else:
                exported = self.export_labels()
                if not exported:
                    return
        
        self._advance_to_next_video_force()

    def prev_video(self):
        if not self.video_files:
            messagebox.showinfo("No Videos", "Open a folder first")
            return
        
        if self.labeled_frames and not self.current_video_exported:
            status = self.get_video_status(self.video_path)
            if status and status.get('completed', False):
                self._advance_to_prev_video_force()
                return
                
            if not messagebox.askyesno("Export Labels First", 
                                    f"You have {len(self.labeled_frames)} labeled frames.\nExport before moving?"):
                self._advance_to_prev_video_force()
                return
            else:
                exported = self.export_labels()
                if exported:
                    self._advance_to_prev_video_force()
                return
        
        self._advance_to_prev_video_force()

    def _advance_to_prev_video_force(self):
        if not self.video_files:
            return
        
        self.stop_playback()
        
        if self.labeled_frames and self.video_path:
            self.save_frame_progress()
        
        if self.current_video_idx > 0:
            self.current_video_idx -= 1
            self.load_video(str(self.video_files[self.current_video_idx]))
            self.status_var.set(f"⏮ Moved to: {os.path.basename(self.video_files[self.current_video_idx])}")
        else:
            messagebox.showinfo("Start of Folder", "You're at the first video!")
    
    def show_progress(self):
        if not self.video_files:
            messagebox.showinfo("No Videos", "Open a folder first")
            return
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("📊 Labeling Progress")
        progress_window.geometry("700x500")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        scrollbar = tk.Scrollbar(progress_window)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(progress_window, yscrollcommand=scrollbar.set, font=('Courier', 10))
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.config(command=listbox.yview)
        
        listbox.insert(tk.END, f"{'Status':<8} {'Video Name':<50} {'Frames':<15} {'Date':<20}")
        listbox.insert(tk.END, "=" * 95)
        
        completed_count = 0
        for video_path in self.video_files:
            video_name = os.path.basename(video_path)
            status = self.get_video_status(video_path)
            
            if status and status.get('completed', False):
                frames = status.get('frames_labeled', 0)
                total = status.get('total_frames', 0)
                date = status.get('completion_date', '')[:10]
                listbox.insert(tk.END, f"{'✅ DONE':<8} {video_name:<50} {frames}/{total:<10} {date:<20}")
                completed_count += 1
            elif status and not status.get('completed', False):
                frames = status.get('frames_labeled', 0)
                total = status.get('total_frames', 0)
                listbox.insert(tk.END, f"{'🔄 IN PROGRESS':<8} {video_name:<50} {frames}/{total:<10} {'':<20}")
            else:
                listbox.insert(tk.END, f"{'⬜ PENDING':<8} {video_name:<50} {'0/0':<15} {'':<20}")
        
        total = len(self.video_files)
        percent = (completed_count/total*100) if total > 0 else 0
        
        listbox.insert(tk.END, "=" * 95)
        listbox.insert(tk.END, f"📊 Completed: {completed_count}/{total} videos ({percent:.1f}%)")
        
        btn_frame = ttk.Frame(progress_window)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Close", command=progress_window.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🔄 Reset Progress", 
                  command=lambda: self.reset_progress(progress_window)).pack(side=tk.LEFT, padx=5)
        
        progress_window.bind('<Escape>', lambda e: progress_window.destroy())
        self.video_panel.focus_set()
    
    def reset_progress(self, window):
        if messagebox.askyesno("Reset Progress", "Reset ALL progress?"):
            self.progress_data = {}
            self.save_progress()
            self.update_video_progress()
            window.destroy()
            self.status_var.set("🔄 Progress reset")
            messagebox.showinfo("Reset Complete", "All progress has been reset.")
    
    # ============ EXPORT ============
    
    def export_labels(self):
        if not self.labeled_frames:
            messagebox.showwarning("No Labels", "No frames labeled")
            return False
        
        video_name = os.path.splitext(os.path.basename(self.video_path))[0] if self.video_path else "labels"
        default_filename = f"{video_name}_labels.json"
        
        export_path = filedialog.asksaveasfilename(
            title="Export Labels",
            defaultextension=".json",
            initialfile=default_filename,
            filetypes=[("JSON files", "*.json")]
        )
        
        if not export_path:
            return False
        
        # Labels are stored in DISPLAY space (video resized to 600px tall in
        # show_frame). train_yolo.py normalizes by the NATIVE frame size, so we
        # convert display -> native video pixels here. inv_scale = video_h / 600.
        vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else 0
        vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else 0
        disp_h = 600
        inv_scale = (vid_h / disp_h) if vid_h else 1.0

        export_keyframes = []
        for f in self.labeled_frames:
            fc = dict(f)
            fc['points'] = {k: [int(round(x * inv_scale)), int(round(y * inv_scale))]
                            for k, (x, y) in f['points'].items()}
            export_keyframes.append(fc)

        data = {
            'video': os.path.basename(self.video_path) if self.video_path else "unknown",
            'total_frames': self.total_frames,
            'fps': self.fps,
            'frame_width': vid_w,
            'frame_height': vid_h,
            'coordinate_space': 'video_pixels',
            'keyframes': export_keyframes,
            'keypoint_names': self.keypoint_names,
            'export_date': datetime.now().isoformat(),
            'tracking_method': 'optical_flow + manual',
            'confidence_threshold': self.confidence_threshold
        }
        
        try:
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.status_var.set(f"✅ Exported to: {export_path}")
            self.current_video_exported = True
            
            if self.video_path:
                self.mark_video_complete(self.video_path)
                self.status_var.set(f"✅ Exported and marked complete!")
            
            if self.video_files and len(self.video_files) > 1:
                if self.is_playing:
                    self.is_playing = False
                    self.play_btn.config(text="▶ Play")
                    if hasattr(self, '_after_id'):
                        self.root.after_cancel(self._after_id)
                
                self._advance_to_next_video_force()
            
            return True
                        
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export:\n{str(e)}")
            self.status_var.set(f"❌ Export failed: {e}")
            return False

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("❌ PIL/Pillow not installed. Run: pip install pillow")
        exit(1)
    
    # Check for ultralytics
    try:
        import ultralytics
        print(f"✅ Ultralytics YOLO version: {ultralytics.__version__}")
    except ImportError:
        print("❌ Ultralytics not installed. Run: pip install ultralytics")
        # Continue anyway - app will show error when trying to use YOLO
    
    root = tk.Tk()
    app = VideoLabelerGUI(root)
    root.mainloop()