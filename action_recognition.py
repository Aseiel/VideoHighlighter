import cv2
import numpy as np
from pathlib import Path
from openvino.runtime import Core
import csv
import json
import threading
import time
import argparse
from collections import Counter, deque

# =============================
# Load labels - support both Kinetics-400 and custom models
# =============================
BASE_DIR = Path(__file__).parent.resolve()

# Try to load custom model labels first, fall back to Kinetics-400
CUSTOM_MAPPING_PATH = BASE_DIR / "intel_finetuned_classifier_3d_mapping.json"
KINETICS_LABELS_PATH = BASE_DIR / "kinetics_400_labels.json"

# Global variables for label mappings
CUSTOM_LABELS = None
KINETICS_400_LABELS = None

def load_label_mappings():
    global CUSTOM_LABELS, KINETICS_400_LABELS
    
    # Load custom model labels
    if CUSTOM_MAPPING_PATH.exists():
        with open(CUSTOM_MAPPING_PATH, "r") as f:
            custom_data = json.load(f)
            CUSTOM_LABELS = custom_data.get('idx_to_label', {})
            # Convert string keys to integers
            CUSTOM_LABELS = {int(k): v for k, v in CUSTOM_LABELS.items()}
        print(f"✓ Loaded custom model labels: {len(CUSTOM_LABELS)} classes")
    else:
        CUSTOM_LABELS = None
    
    # Load Kinetics-400 labels
    if KINETICS_LABELS_PATH.exists():
        with open(KINETICS_LABELS_PATH, "r") as f:
            KINETICS_400_LABELS = json.load(f)
        print(f"✓ Loaded Kinetics-400 labels: {len(KINETICS_400_LABELS)} classes")
    else:
        print("⚠️ Kinetics-400 labels not found")

# Call this at startup
load_label_mappings()

def get_action_name(action_id):
    """Get action name - try custom labels first, then Kinetics-400"""
    if CUSTOM_LABELS and action_id in CUSTOM_LABELS:
        return CUSTOM_LABELS[action_id]
    elif KINETICS_400_LABELS:
        return KINETICS_400_LABELS.get(str(action_id), f"action_{action_id}")
    else:
        return f"action_{action_id}"

def get_id_from_name(name):
    """Get action ID - try custom labels first, then Kinetics-400"""
    # Try custom labels
    if CUSTOM_LABELS:
        for idx, action_name in CUSTOM_LABELS.items():
            if action_name.lower() == name.lower():
                return idx
            
    # Try Kinetics-400
    if KINETICS_400_LABELS:
        for k, v in KINETICS_400_LABELS.items():
            if v.lower() == name.lower():
                return int(k)
    
    raise ValueError(f"Action '{name}' not found in any label set")

# =============================
# Model paths
# =============================
ENCODER_XML = BASE_DIR / "models/intel_action/encoder/FP32/action-recognition-0001-encoder.xml"
ENCODER_BIN = BASE_DIR / "models/intel_action/encoder/FP32/action-recognition-0001-encoder.bin"
SEQUENCE_LENGTH = 16

# =============================
# Visualization settings
# =============================
ACTION_LABEL_BG_COLOR = (0, 0, 0)  # Black background
ACTION_LABEL_TEXT_COLOR = (0, 255, 255)  # Cyan text
ACTION_LABEL_HIGH_CONF_COLOR = (0, 255, 0)  # Green for high confidence
ACTION_FONT_SCALE = 0.8
ACTION_FONT_THICKNESS = 2
ACTION_LABEL_ALPHA = 0.7  # Semi-transparent background

# =============================
# Async Inference Engine with GUI Stats
# =============================
class AsyncBatchedInferenceEngine:
    def __init__(self, compiled_model, input_layer, output_layer, num_requests=2):
        self.compiled_model = compiled_model
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.num_requests = num_requests
        self.requests = [compiled_model.create_infer_request() for _ in range(num_requests)]
        self.current_request = 0
        self.total_inferences = 0
        self.start_time = time.time()

    def infer_async(self, frame):
        request = self.requests[self.current_request]
        self.current_request = (self.current_request + 1) % self.num_requests
        request.start_async({self.input_layer.any_name: frame})
        self.total_inferences += 1
        return request

    def wait_and_get(self, request):
        request.wait()
        return request.get_tensor(self.output_layer).data

    def get_stats(self):
        elapsed = time.time() - self.start_time
        fps = self.total_inferences / elapsed if elapsed > 0 else 0
        return {
            'total_inferences': self.total_inferences,
            'elapsed_time': elapsed,
            'inference_fps': fps
        }

# =============================
# Load models
# =============================
def load_models(device="AUTO"):
    ie = Core()
    available_devices = ie.available_devices
    print(f"Available OpenVINO devices: {available_devices}")

    if device == "AUTO":
        device_priority = ["GPU.1", "GPU.0", "GPU", "CPU"]
        selected_device = next((d for d in device_priority if d in available_devices), "CPU")
    else:
        selected_device = device if device in available_devices else "CPU"

    print(f"Using device: {selected_device}")

    # Encoder (always required) 
    if not ENCODER_XML.exists() or not ENCODER_BIN.exists():
        raise FileNotFoundError(f"❌ Encoder model not found at {ENCODER_XML}")
    print("✓ Encoder model found")

    encoder_model = ie.read_model(model=ENCODER_XML, weights=ENCODER_BIN)
    compiled_encoder = ie.compile_model(model=encoder_model, device_name=selected_device)

    # Decoder
    custom_decoder_xml = BASE_DIR / "action_classifier_3d.xml"
    custom_decoder_bin = BASE_DIR / "action_classifier_3d.bin"

    if custom_decoder_xml.exists() and custom_decoder_bin.exists():
        print("✓ Using custom fine-tuned decoder model")
        decoder_model_path = custom_decoder_xml
        decoder_weights_path = custom_decoder_bin
    else:
        print("⚠️ Custom model not found, falling back to Intel Kinetics-400 decoder")
        decoder_model_path = BASE_DIR / "models/intel_action/decoder/FP32/action-recognition-0001-decoder.xml"
        decoder_weights_path = BASE_DIR / "models/intel_action/decoder/FP32/action-recognition-0001-decoder.bin"

        if not decoder_model_path.exists() or not decoder_weights_path.exists():
            raise FileNotFoundError(f"❌ Neither custom nor Intel default decoder found!")

    decoder_model = ie.read_model(model=decoder_model_path, weights=decoder_weights_path)
    compiled_decoder = ie.compile_model(model=decoder_model, device_name=selected_device)

    print("✓ Model loading complete.\n")

    return (
        compiled_encoder, compiled_encoder.input(0), compiled_encoder.output(0),
        compiled_decoder, compiled_decoder.input(0), compiled_decoder.output(0),
        selected_device
    )

# =============================
# Preprocess frame
# =============================
def preprocess_frame(frame, input_shape):
    N, C, H, W = input_shape
    h, w = frame.shape[:2]
    scale = min(W / w, H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    frame_resized = cv2.resize(frame, (new_w, new_h))
    pad_top = (H - new_h) // 2
    pad_bottom = H - new_h - pad_top
    pad_left = (W - new_w) // 2
    pad_right = W - new_w - pad_left
    frame_padded = cv2.copyMakeBorder(frame_resized, pad_top, pad_bottom, pad_left, pad_right,
                                      borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
    frame_padded = frame_padded.transpose(2, 0, 1)
    return np.expand_dims(frame_padded, axis=0).astype(np.float32)

# =============================
# Draw action labels on frame
# =============================
def draw_action_labels(frame, detected_actions, max_labels=3):
    """
    Draw detected action labels on frame
    
    Args:
        frame: Input frame
        detected_actions: List of (action_name, score) tuples, sorted by score
        max_labels: Maximum number of labels to display
    
    Returns:
        Annotated frame
    """
    if not detected_actions:
        return frame
    
    annotated = frame.copy()
    h, w = frame.shape[:2]
    
    # Limit to top actions
    top_actions = detected_actions[:max_labels]
    
    # Calculate panel size
    panel_height = 30 + (len(top_actions) * 35)
    panel_width = 400
    panel_x = w - panel_width - 10
    panel_y = 10
    
    # Create semi-transparent overlay
    overlay = annotated.copy()
    cv2.rectangle(overlay, 
                  (panel_x, panel_y), 
                  (panel_x + panel_width, panel_y + panel_height),
                  ACTION_LABEL_BG_COLOR, 
                  -1)
    
    # Blend overlay with original
    cv2.addWeighted(overlay, ACTION_LABEL_ALPHA, annotated, 1 - ACTION_LABEL_ALPHA, 0, annotated)
    
    # Draw border
    cv2.rectangle(annotated,
                  (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  ACTION_LABEL_TEXT_COLOR,
                  2)
    
    # Draw title
    cv2.putText(annotated, "DETECTED ACTIONS", 
                (panel_x + 10, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                ACTION_LABEL_TEXT_COLOR, 
                2)
    
    # Draw each action
    y_offset = panel_y + 50
    for i, (action_name, score) in enumerate(top_actions):
        # Choose color based on confidence
        text_color = ACTION_LABEL_HIGH_CONF_COLOR if score > 0.5 else ACTION_LABEL_TEXT_COLOR
        
        # Format action text
        action_text = f"{i+1}. {action_name}"
        score_text = f"{score:.2%}"
        
        # Draw action name
        cv2.putText(annotated, action_text,
                    (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    1)
        
        # Draw score
        cv2.putText(annotated, score_text,
                    (panel_x + panel_width - 70, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    1)
        
        # Draw confidence bar
        bar_width = int((panel_width - 30) * score)
        cv2.rectangle(annotated,
                      (panel_x + 10, y_offset + 5),
                      (panel_x + 10 + bar_width, y_offset + 10),
                      text_color,
                      -1)
        
        y_offset += 35
    
    return annotated

# =============================
# Run action detection
# =============================
def run_action_detection(video_path, device="AUTO", sample_rate=5, log_file="action_log.csv",
                         debug=False, top_k=50, confidence_threshold=0.01, show_video=False,
                         num_requests=2, interesting_actions=None,
                         progress_callback=None, cancel_flag=None,
                         draw_labels=False, annotated_output=None):
    """
    Run action recognition on video
    
    Args:
        draw_labels: If True, draw action labels on frames
        annotated_output: Path for annotated video output
    """
    if interesting_actions is not None:
        interesting_actions_set = set([s.lower() for s in interesting_actions])
    else:
        interesting_actions_set = None

    compiled_encoder, encoder_input, encoder_output, compiled_decoder, decoder_input, decoder_output, actual_device = load_models(device)
    encoder_engine = AsyncBatchedInferenceEngine(compiled_encoder, encoder_input, encoder_output, num_requests=num_requests)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_processed_frames = total_frames // sample_rate
    
    # Setup video writer for annotated output
    video_writer = None
    if draw_labels and annotated_output:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(annotated_output, fourcc, fps, (frame_width, frame_height))
        print(f"🎨 Creating annotated video: {annotated_output}")

    sequence_buffer = []
    all_actions = []
    prev_req = None
    frame_id = 0
    processed_frames = 0
    detection_count = 0
    
    # For visualization: keep track of recent detections
    recent_detections = deque(maxlen=SEQUENCE_LENGTH)

    start_time = time.time()
    last_gui_update = start_time

    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_mmss", "frame_id", "action_id", "action_name", "score", "timestamp_seconds"])

        while True:
            if cancel_flag and cancel_flag.is_set():
                print("⚠️ Action detection canceled by user.")
                break

            ret, frame = cap.read()
            if not ret:
                break
            
            original_frame = frame.copy() if draw_labels else None
            frame_id += 1
            
            if frame_id % sample_rate != 0:
                # Still write frame to annotated video if enabled
                if video_writer and draw_labels and original_frame is not None:
                    # Draw last known detections
                    if recent_detections:
                        annotated = draw_action_labels(original_frame, list(recent_detections))
                        video_writer.write(annotated)
                    else:
                        video_writer.write(original_frame)
                continue

            timestamp_secs = frame_id / fps
            mins, secs = divmod(int(timestamp_secs), 60)
            timestamp_str = f"{mins:02d}:{secs:02d}"

            processed_frame = preprocess_frame(frame, encoder_input.shape)
            req = encoder_engine.infer_async(processed_frame)

            if prev_req is not None:
                features = encoder_engine.wait_and_get(prev_req)[0]
                features = np.reshape(features, (-1,))
                sequence_buffer.append(features)
                if len(sequence_buffer) > SEQUENCE_LENGTH:
                    sequence_buffer.pop(0)

                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    sequence_array = np.expand_dims(np.stack(sequence_buffer, axis=0), axis=0)
                    predictions = compiled_decoder([sequence_array])[decoder_output].flatten()
                    
                    # Clear recent detections for this frame
                    frame_detections = []

                    if interesting_actions_set:
                        # Check only the requested actions
                        for action_name in interesting_actions_set:
                            action_id = get_id_from_name(action_name)
                            score = float(predictions[action_id])
                            if score >= confidence_threshold:
                                writer.writerow([timestamp_str, frame_id, action_id, action_name, score, timestamp_secs])
                                all_actions.append((timestamp_secs, frame_id, action_id, score, action_name))
                                frame_detections.append((action_name, score))
                                detection_count += 1
                                if debug:
                                    print(f"{timestamp_str} -> {action_name} (score:{score:.3f})")
                    else:
                        # fallback to top-k
                        top_indices = np.argsort(predictions)[-top_k:][::-1]
                        for idx in top_indices:
                            score = float(predictions[idx])
                            action_name = get_action_name(idx)
                            if score >= confidence_threshold:
                                writer.writerow([timestamp_str, frame_id, idx, action_name, score, timestamp_secs])
                                all_actions.append((timestamp_secs, frame_id, idx, score, action_name))
                                frame_detections.append((action_name, score))
                                detection_count += 1
                                if debug:
                                    print(f"{timestamp_str} -> {action_name} (score:{score:.3f})")
                    
                    # Update recent detections (sorted by score)
                    if frame_detections:
                        frame_detections.sort(key=lambda x: x[1], reverse=True)
                        recent_detections.clear()
                        recent_detections.extend(frame_detections[:3])  # Keep top 3

            # Draw labels and write annotated frame
            if video_writer and draw_labels and original_frame is not None:
                if recent_detections:
                    annotated = draw_action_labels(original_frame, list(recent_detections))
                    video_writer.write(annotated)
                else:
                    video_writer.write(original_frame)

            prev_req = req
            processed_frames += 1

            current_time = time.time()
            if progress_callback and (current_time - last_gui_update > 0.1):
                elapsed = current_time - start_time
                processing_fps = processed_frames / elapsed if elapsed > 0 else 0
                engine_stats = encoder_engine.get_stats()
                progress_msg = (f"Frame {processed_frames}/{expected_processed_frames} | "
                                f"Detections: {detection_count} | "
                                f"Processing: {processing_fps:.1f} FPS | "
                                f"Inference: {engine_stats['inference_fps']:.1f} FPS | "
                                f"Device: {actual_device}")
                progress_callback(processed_frames, expected_processed_frames, "Enhanced Action Recognition", progress_msg)
                last_gui_update = current_time

        # Flush last frame
        if prev_req is not None:
            features = encoder_engine.wait_and_get(prev_req)[0]
            features = np.reshape(features, (-1,))
            sequence_buffer.append(features)
            if len(sequence_buffer) > SEQUENCE_LENGTH:
                sequence_buffer.pop(0)

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                sequence_array = np.expand_dims(np.stack(sequence_buffer, axis=0), axis=0)
                predictions = compiled_decoder([sequence_array])[decoder_output].flatten()
                top_indices = np.argsort(predictions)[-top_k:][::-1]

                for idx in top_indices:
                    score = float(predictions[idx])
                    action_name = get_action_name(idx)
                    if score >= confidence_threshold:
                        if interesting_actions_set is None or action_name.lower() in interesting_actions_set:
                            writer.writerow([timestamp_str, frame_id, idx, action_name, score, timestamp_secs])
                            all_actions.append((timestamp_secs, frame_id, idx, score, action_name))
                            detection_count += 1
                            if debug:
                                print(f"{timestamp_str} -> {action_name} (score:{score:.3f})")

    cap.release()
    if video_writer:
        video_writer.release()
        print(f"✅ Annotated video saved: {annotated_output}")

    if progress_callback:
        total_time = time.time() - start_time
        engine_stats = encoder_engine.get_stats()
        final_msg = (f"Complete! {detection_count} actions detected | "
                     f"Processed {processed_frames} frames in {total_time:.1f}s | "
                     f"Avg Processing: {processed_frames/total_time:.1f} FPS | "
                     f"Avg Inference: {engine_stats['inference_fps']:.1f} FPS")
        progress_callback(processed_frames, expected_processed_frames, "Action Recognition Complete", final_msg)

    return all_actions

# =============================
# Debug / Analysis Functions
# =============================
def print_top_actions(all_actions, top_n=20):
    sorted_actions = sorted(all_actions, key=lambda x: x[3], reverse=True)
    print(f"\nTop {min(top_n, len(sorted_actions))} actions (by confidence):")
    for i, (timestamp, frame_id, action_id, score, action_name) in enumerate(sorted_actions[:top_n]):
        mins, secs = divmod(int(timestamp), 60)
        print(f"{i+1:2d}. {mins:02d}:{secs:02d} -> {action_name} (score:{score:.3f})")

def print_most_common_actions(all_actions, top_n=20):
    counter = Counter([a[4] for a in all_actions])
    print(f"\nTop {min(top_n, len(counter))} most common actions:")
    for i, (action_name, count) in enumerate(counter.most_common(top_n)):
        print(f"{i+1:2d}. {action_name} ({count} occurrences)")

def detect_action_sequences(all_actions, score_threshold=0.01, min_duration=1.0):
    sequences = []
    current_seq = None
    for timestamp, frame_id, action_id, score, action_name in all_actions:
        if score < score_threshold:
            if current_seq:
                current_seq['end_time'] = timestamp
                sequences.append(current_seq)
                current_seq = None
            continue
        if current_seq and current_seq['action_name'] == action_name:
            current_seq['max_score'] = max(current_seq['max_score'], score)
            current_seq['end_time'] = timestamp
        else:
            if current_seq:
                sequences.append(current_seq)
            current_seq = {'action_name': action_name, 'start_time': timestamp,
                           'end_time': timestamp, 'max_score': score}
    if current_seq:
        sequences.append(current_seq)
    sequences = [seq for seq in sequences if (seq['end_time'] - seq['start_time']) >= min_duration]
    return sequences

def print_action_sequences(all_actions):
    sequences = detect_action_sequences(all_actions)
    print(f"\nAction sequences detected ({len(sequences)}):")
    for i, seq in enumerate(sequences):
        duration = seq['end_time'] - seq['start_time']
        start_mins, start_secs = divmod(int(seq['start_time']), 60)
        end_mins, end_secs = divmod(int(seq['end_time']), 60)
        print(f"{i+1:2d}. {seq['action_name']} Duration: {duration:.1f}s "
              f"({start_mins:02d}:{start_secs:02d} - {end_mins:02d}:{end_secs:02d}) "
              f"Max score: {seq['max_score']:.3f}")

# =============================
# CLI
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--device", type=str, default="AUTO")
    parser.add_argument("--sample-rate", type=int, default=30)
    parser.add_argument("--log-file", type=str, default="action_log.csv")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show-video", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--draw-labels", action="store_true", help="Draw action labels on frames")
    parser.add_argument("--annotated-output", type=str, help="Output path for annotated video")
    args = parser.parse_args()

    results = run_action_detection(
        video_path=args.input,
        device=args.device,
        sample_rate=args.sample_rate,
        log_file=args.log_file,
        debug=args.debug,
        top_k=args.top_k,
        confidence_threshold=args.confidence,
        show_video=args.show_video,
        draw_labels=args.draw_labels,
        annotated_output=args.annotated_output
    )
    
    print_top_actions(results)
    print_most_common_actions(results)
    print_action_sequences(results)
    print(f"\n✅ Processing complete. Found {len(results)} action detections.")