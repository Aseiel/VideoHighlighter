# modules/video_cache.py
"""
Video Analysis Cache Module
Single-class implementation: VideoAnalysisCache

- Keeps original analysis cache API: exists/save/load/invalidate/clear_all/list_cached_videos/get_cache_info
- Adds highlight cache API: save_highlight_segments/load_highlight_segments/get_highlight_history/get_cache_stats
- Thread-safe, atomic writes for highlight cache
"""

import json
import hashlib
import os
import time
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import threading
import shutil


# ========== DATA CLASSES (unchanged external API) ==========

def build_analysis_cache_params(gui_config: dict, config: dict, sample_rate: int, video_duration: float):
    # “Analysis params” = anything that changes the computed analysis artifacts
    # Keep values JSON-serializable and stable (sort lists)
    highlight_objects = gui_config.get("highlight_objects", config.get("highlight_objects", [])) or []
    interesting_actions = gui_config.get("interesting_actions", []) or []
    search_keywords = gui_config.get("search_keywords", []) or []

    # Time range settings (if enabled)
    use_time_range = bool(gui_config.get("use_time_range", False))
    range_start = int(gui_config.get("range_start", 0) or 0)
    range_end = gui_config.get("range_end", None)
    range_end = int(range_end) if range_end is not None else None

    # YOLO settings
    yolo_model_size = str(gui_config.get("yolo_model_size") or "n").lower()
    openvino_model_folder = gui_config.get("openvino_model_folder", f"yolo11{yolo_model_size}_openvino_model/")
    yolo_pt_path = gui_config.get("yolo_pt_path", f"yolo11{yolo_model_size}.pt")

    params = {
        # bump this when you change the meaning/format of cached analysis
        "analysis_cache_schema": "analysis_v2",

        # core toggles
        "use_transcript": bool(gui_config.get("use_transcript", False)),
        "transcript_model": str(gui_config.get("transcript_model", "medium")),
        "search_keywords": sorted([str(k).lower() for k in search_keywords]),

        "highlight_objects": sorted([str(o) for o in highlight_objects]),
        "interesting_actions": sorted([str(a) for a in interesting_actions]),

        # object/action sampling knobs
        "object_frame_skip": int(gui_config.get("object_frame_skip", gui_config.get("clip_time", 10) or 10)),
        "sample_rate": int(sample_rate),

        # action detector knobs used in your call
        "action_use_person_detection": True,
        "action_max_people": int(gui_config.get("action_max_people", 2) or 2),

        # yolo identity
        "yolo_model_size": yolo_model_size,
        "yolo_pt_path": str(yolo_pt_path),
        "openvino_model_folder": str(openvino_model_folder),

        # time-range
        "use_time_range": use_time_range,
        "range_start": range_start if use_time_range else 0,
        "range_end": range_end if use_time_range else None,

        # optional: points affect scoring, not analysis — but if you cache “analysis only”
        # you can omit scoring params. If you cache waveforms/peaks based on thresholds,
        # include them.
        "scene_threshold": float(gui_config.get("scene_threshold", 70.0)),
        "motion_threshold": float(gui_config.get("motion_threshold", 100.0)),
        "spike_factor": float(gui_config.get("spike_factor", 1.2)),
        "freeze_seconds": float(gui_config.get("freeze_seconds", 4)),
        "freeze_factor": float(gui_config.get("freeze_factor", 0.8)),
    }
    return params


def atomic_write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent)
    )
    tmp_path = Path(tmp_path)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            # Use custom encoder to handle numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    # Handle numpy integers
                    if hasattr(obj, 'dtype') and hasattr(obj, 'item'):
                        return obj.item()
                    # Handle numpy floats
                    elif hasattr(obj, 'dtype') and hasattr(obj, 'tolist'):
                        return obj.tolist() if hasattr(obj, 'shape') and len(obj.shape) > 0 else float(obj)
                    # Handle numpy arrays
                    elif hasattr(obj, 'tolist'):
                        return obj.tolist()
                    # Let the base class default method raise the TypeError
                    return super().default(obj)
            
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(path))  # atomic on same filesystem
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise


@dataclass
class HighlightSegment:
    """A single highlight segment with metadata"""
    start_time: float
    end_time: float
    duration: float
    score: float = 0.0
    selected_at: str = field(default_factory=lambda: datetime.now().isoformat())
    signals: Dict[str, float] = field(default_factory=dict)
    primary_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HighlightSegment":
        return cls(**data)


@dataclass
class HighlightMetadata:
    """Metadata for a highlight generation run"""
    video_path: str
    video_hash: str
    parameters_hash: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_duration: float = 0.0
    target_duration: float = 0.0
    duration_mode: str = "MAX"
    segments_count: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    score_distribution: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HighlightMetadata":
        return cls(**data)


class CachedAnalysisData:
    """
    Backward-compatible wrapper around analysis cache dict
    """

    def __init__(self, cache_data: Dict[str, Any]):
        self.video_path = cache_data.get("video_path")
        self.video_hash = cache_data.get("video_hash")
        self.cached_at = cache_data.get("cached_at")

        self.video_metadata = cache_data.get("video_metadata", {})
        self.duration = self.video_metadata.get("duration", 0)
        self.fps = self.video_metadata.get("fps", 30)
        self.resolution = self.video_metadata.get("resolution", "unknown")

        self.transcript = cache_data.get("transcript", {"segments": []})
        self.objects = cache_data.get("objects", [])
        self.actions = cache_data.get("actions", [])
        self.scenes = cache_data.get("scenes", [])
        self.audio_peaks = cache_data.get("audio_peaks", [])
        # your pipeline writes motion_events/motion_peaks (not motion_scores)
        self.motion_events = cache_data.get("motion_events", [])
        self.motion_peaks = cache_data.get("motion_peaks", [])
        # keep legacy field too if present
        self.motion_scores = cache_data.get("motion_scores", [])

    def get_transcript(self, start: float, end: float, 
                                    mode: str = "overlap") -> List[Dict[str, Any]]:
        """
        Get transcript segments in time range.
        
        mode:
            - "overlap": segments that overlap with the range (default)
            - "contained": only segments fully within the range
            - "strict": segments where start >= range_start AND end <= range_end
        """
        segments = self.transcript.get("segments", [])
        
        if mode == "contained":
            return [
                seg for seg in segments 
                if seg.get("start", 0) >= start and seg.get("end", 0) <= end
            ]
        elif mode == "strict":
            return [
                seg for seg in segments
                if start <= seg.get("start", 0) and seg.get("end", 0) <= end
            ]
        else:  # overlap (most permissive)
            return [
                seg for seg in segments
                if not (seg.get("end", 0) < start or seg.get("start", 0) > end)
            ]


    def get_objects_in_timerange(self, start: float, end: float) -> List[Dict[str, Any]]:
        return [obj for obj in self.objects if start <= obj.get("timestamp", 0) <= end]

    def get_actions_in_timerange(self, start: float, end: float) -> List[Dict[str, Any]]:
        return [action for action in self.actions if start <= action.get("timestamp", 0) <= end]

    def get_scenes_in_timerange(self, start: float, end: float) -> List[Dict[str, Any]]:
        return [scene for scene in self.scenes if not (scene.get("end", 0) < start or scene.get("start", 0) > end)]



# ========== SINGLE CACHE CLASS ==========

class VideoAnalysisCache:
    """
    VideoAnalysisCache: analysis cache + highlight segment cache (single class)

    Analysis cache file:  <cache_dir>/<video_hash>.cache.json
    Highlights cache file: <cache_dir>/highlights/<video_hash>/<params_hash>.json
    """

    def __init__(
        self,
        cache_dir: str = "./cache",
        max_cache_size_mb: int = 2048,
        enable_highlight_cache: bool = True,
        max_highlight_versions: int = 10,
    ):
        print(f"\n🔧 [DEBUG] VideoAnalysisCache.__init__")
        print(f"  - cache_dir: {cache_dir}")
        print(f"  - enable_highlight_cache: {enable_highlight_cache}")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Cache directory: {self.cache_dir.absolute()}")

        self.max_cache_size = int(max_cache_size_mb) * 1024 * 1024
        self.enable_highlight_cache = bool(enable_highlight_cache)
        self.max_highlight_versions = int(max_highlight_versions)
        
        # Initialize base_cache as self reference
        self.base_cache = self
        print(f"  ✓ Initialized base_cache = self")

        self._lock = threading.RLock()
        print(f"  ✓ Created thread lock")

        # enhanced directory structure
        (self.cache_dir / "highlights").mkdir(exist_ok=True)
        (self.cache_dir / "temp").mkdir(exist_ok=True)
        print(f"  ✓ Created subdirectories: highlights/, temp/")

        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "highlight_hits": 0,
            "highlight_misses": 0,
        }
        print(f"  ✓ Initialized stats")
        print(f"🔧 [DEBUG] __init__ complete\n")

    # ---------- hashing / paths ----------

    def _get_video_hash(self, video_path: str) -> str:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        stat = video_path.stat()
        hash_string = f"{video_path.absolute()}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def _get_cache_path(self, video_path: str) -> Path:
        video_hash = self._get_video_hash(video_path)
        return self.cache_dir / f"{video_hash}.cache.json"

    def _get_parameters_hash(self, parameters: Dict[str, Any]) -> str:
        # ensure stable hash
        params_str = json.dumps(parameters, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(params_str.encode()).hexdigest()

    def _highlight_dir(self, video_hash: str) -> Path:
        return self.cache_dir / "highlights" / video_hash

    # ---------- analysis cache API (backward compatible) ----------

    def _make_signature(self, params: Dict[str, Any]) -> str:
        payload = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _get_analysis_cache_path_for_signature(self, video_path: str, signature: str) -> Path:
        """
        Signature-based analysis cache path.
        Creates a new cache file when parameters change.
        """
        video_hash = self._get_video_hash(video_path)
        return self.cache_dir / f"{video_hash}.{signature}.cache.json"

    def exists(self, video_path: str, params: Optional[Dict[str, Any]] = None) -> bool:
        with self._lock:
            if params is not None:
                signature = self._make_signature(params)
                return self._get_analysis_cache_path_for_signature(video_path, signature).exists()
            return self._get_cache_path(video_path).exists()

    def save(self, video_path: str, analysis_data: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> None:
        """
        Save analysis cache.
        - Atomic write (no partial cache)
        - If params provided: use signature-based filename so parameter changes create a new cache file
        - If params is None: fall back to legacy path (<video_hash>.cache.json) for backward compatibility
        """
        with self._lock:
            video_hash = self._get_video_hash(video_path)

            if params is not None:
                signature = self._make_signature(params)
                cache_path = self._get_analysis_cache_path_for_signature(video_path, signature)
            else:
                signature = None
                cache_path = self._get_cache_path(video_path)  # legacy <video_hash>.cache.json

            cache_data = {
                "video_path": str(Path(video_path).absolute()),
                "video_hash": video_hash,
                "cached_at": datetime.now().isoformat(),
                "cache_version": "1.1",
                "cache_complete": True,
                "analysis_signature": signature,
                "analysis_parameters": params,
                **analysis_data,
            }

            atomic_write_json(cache_path, cache_data)

            self.stats["saves"] += 1
            print(f"✓ Cache saved: {cache_path}")

    def load(self, video_path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Load analysis cache.
        - If params provided: load signature-based cache (new cache per parameter change)
        - If params is None: fall back to legacy cache path
        """
        with self._lock:
            if params is not None:
                signature = self._make_signature(params)
                cache_path = self._get_analysis_cache_path_for_signature(video_path, signature)
            else:
                signature = None
                cache_path = self._get_cache_path(video_path)

            if not cache_path.exists():
                self.stats["misses"] += 1
                return None

            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                current_hash = self._get_video_hash(video_path)
                if cache_data.get("video_hash") != current_hash:
                    print("⚠ Cache is outdated (video file changed), will re-process")
                    self.stats["misses"] += 1
                    return None

                if cache_data.get("cache_complete") is not True:
                    print("⚠ Cache incomplete, will re-process")
                    self.stats["misses"] += 1
                    return None

                # Extra safety: if params were passed, ensure signature matches too
                if params is not None and cache_data.get("analysis_signature") != signature:
                    self.stats["misses"] += 1
                    return None

                print(f"✓ Cache loaded from: {cache_path}")
                self.stats["hits"] += 1
                return cache_data

            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠ Cache file corrupted: {e}, will re-process")
                self.stats["misses"] += 1
                return None

    # convenience aliases (optional usage in your pipeline)
    def save_enhanced(self, video_path: str, analysis_data: Dict[str, Any]) -> bool:
        with self._lock:
            try:
                self.save(video_path, analysis_data)
                return True
            except Exception as e:
                print(f"❌ Enhanced save failed: {e}")
                return False

    def load_enhanced(self, video_path: str) -> Optional[Dict[str, Any]]:
        return self.load(video_path)

    def invalidate(self, video_path: str) -> bool:
        with self._lock:
            cache_path = self._get_cache_path(video_path)
            if cache_path.exists():
                cache_path.unlink()
                print(f"✓ Cache deleted: {cache_path}")
                return True
            return False

    def clear_all(self) -> int:
        with self._lock:
            count = 0
            for cache_file in self.cache_dir.glob("*.cache.json"):
                cache_file.unlink()
                count += 1
            print(f"✓ Cleared {count} cache file(s)")
            return count

    def list_cached_videos(self) -> List[Dict[str, Any]]:
        with self._lock:
            cached_videos: List[Dict[str, Any]] = []
            for cache_file in self.cache_dir.glob("*.cache.json"):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)

                    cached_videos.append(
                        {
                            "video_path": cache_data.get("video_path", "unknown"),
                            "cached_at": cache_data.get("cached_at", "unknown"),
                            "cache_file": str(cache_file),
                            "video_metadata": cache_data.get("video_metadata", {}),
                        }
                    )
                except (json.JSONDecodeError, KeyError):
                    continue
            return cached_videos

    def get_cache_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cache_path = self._get_cache_path(video_path)
            if not cache_path.exists():
                return None

            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                # pipeline uses motion_events/motion_peaks, but keep legacy key too
                return {
                    "video_path": cache_data.get("video_path"),
                    "cached_at": cache_data.get("cached_at"),
                    "cache_version": cache_data.get("cache_version"),
                    "has_transcript": "transcript" in cache_data,
                    "has_objects": "objects" in cache_data,
                    "has_actions": "actions" in cache_data,
                    "has_scenes": "scenes" in cache_data,
                    "has_audio_peaks": "audio_peaks" in cache_data,
                    "has_motion_events": "motion_events" in cache_data,
                    "has_motion_peaks": "motion_peaks" in cache_data,
                    "has_motion_scores": "motion_scores" in cache_data,
                    "video_metadata": cache_data.get("video_metadata", {}),
                    "cache_size_mb": cache_path.stat().st_size / (1024 * 1024),
                }
            except (json.JSONDecodeError, KeyError):
                return None

    # ---------- highlight segments cache ----------

# modules/video_cache.py

    def save_highlight_segments(self, video_path, parameters, segments, segments_metadata, score_info, analysis_params=None):
        """Save highlight segments with metadata for history
        
        Args:
            video_path: Path to video file
            parameters: Highlight generation parameters
            segments: List of (start, end) segment tuples
            segments_metadata: Metadata for each segment
            score_info: Overall scoring information
            analysis_params: Analysis parameters used to generate the cache (for signature-based loading)
        """
        if not self.enable_highlight_cache:
            print(f"  ❌ enable_highlight_cache is False, returning False")
            return False
        
        try:
            # Create the highlight history entry
            history_entry = {
                'segments': segments,
                'segments_count': len(segments),
                'total_duration': sum(end - start for start, end in segments),
                'parameters': parameters,
                'segments_metadata': segments_metadata,
                'score_info': score_info,
                'created_at': str(__import__('datetime').datetime.now())
            }
            print(f"  ✓ Created history_entry")
            
            # Get existing data or create new - USE self.load() instead of self.base_cache.load()
            print(f"  🔄 Loading existing cache data for {video_path}")
            cache_data = self.load(video_path, params=analysis_params) or {}
            print(f"  ✓ Loaded cache_data with keys: {cache_data.keys() if cache_data else 'empty dict'}")
            
            # Initialize or update highlight history
            if 'highlight_history' not in cache_data:
                print(f"  📝 Initializing new highlight_history list")
                cache_data['highlight_history'] = []
            else:
                print(f"  📊 Existing highlight_history has {len(cache_data['highlight_history'])} entries")
            
            # Add new entry at the beginning
            cache_data['highlight_history'].insert(0, history_entry)
            print(f"  ✓ Added new history entry at position 0")
            
            # Keep only last 10 entries
            old_len = len(cache_data['highlight_history'])
            cache_data['highlight_history'] = cache_data['highlight_history'][:self.max_highlight_versions]
            new_len = len(cache_data['highlight_history'])
            if old_len != new_len:
                print(f"  ✂️ Trimmed history from {old_len} to {new_len} entries")
            
            # Also save as current highlight segments (for backward compatibility)
            cache_data['highlight_segments'] = segments
            cache_data['highlight_metadata'] = {
                'parameters': parameters,
                'segments_metadata': segments_metadata,
                'score_info': score_info,
                'created_at': history_entry['created_at']
            }
            print(f"  ✓ Updated current highlight_segments and highlight_metadata")
            
            # Save back to cache - USE self.save() instead of self.base_cache.save()
            print(f"  💾 Saving updated cache data...")
            self.save(video_path, cache_data, params=analysis_params)
            print(f"  ✅ Save completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Exception in save_highlight_segments: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def load_highlight_segments(
        self, video_path: str, parameters: Dict[str, Any]
    ) -> Optional[Tuple[HighlightMetadata, List[HighlightSegment]]]:
        if not self.enable_highlight_cache:
            self.stats["highlight_misses"] += 1
            return None

        with self._lock:
            try:
                video_hash = self._get_video_hash(video_path)
                params_hash = self._get_parameters_hash(parameters)

                highlight_path = self._highlight_dir(video_hash) / f"{params_hash}.json"
                if not highlight_path.exists():
                    self.stats["highlight_misses"] += 1
                    return None

                with open(highlight_path, "r", encoding="utf-8") as f:
                    highlight_data = json.load(f)

                metadata = HighlightMetadata.from_dict(highlight_data.get("metadata", {}) or {})
                segments = [HighlightSegment.from_dict(d) for d in (highlight_data.get("segments", []) or [])]

                self._update_highlight_access(highlight_path)

                self.stats["highlight_hits"] += 1
                return metadata, segments

            except Exception as e:
                print(f"⚠️ Highlight cache load error: {e}")
                self.stats["highlight_misses"] += 1
                return None

    def _cleanup_old_highlights(self, video_hash: str) -> None:
        hl_dir = self._highlight_dir(video_hash)
        if not hl_dir.exists():
            return

        files = [(p.stat().st_mtime, p) for p in hl_dir.glob("*.json")]
        files.sort(key=lambda x: x[0])  # oldest first

        while len(files) > self.max_highlight_versions:
            _, oldest = files.pop(0)
            try:
                oldest.unlink()
            except Exception:
                pass

    def _update_highlight_access(self, highlight_path: Path) -> None:
        try:
            now = time.time()
            os.utime(highlight_path, (now, now))
        except Exception:
            pass

    # ---------- stats / history ----------

    def get_highlight_history(self, video_path, analysis_params=None):
        """Get history of highlight versions for a video
        
        Args:
            video_path: Path to video file
            analysis_params: Analysis parameters for signature-based cache lookup
        """
        print(f"\n🔍 [DEBUG] get_highlight_history called for: {video_path}")
        history = []
        
        try:
            video_hash = self._get_video_hash(video_path)
            cache_dir = Path(self.cache_dir)
            
            print(f"  - Video hash: {video_hash}")
            print(f"  - analysis_params provided: {analysis_params is not None}")
            
            # Method 1: If params provided, try exact signature match first
            if analysis_params is not None:
                signature = self._make_signature(analysis_params)
                exact_cache_path = self._get_analysis_cache_path_for_signature(video_path, signature)
                print(f"  - Looking for exact signature cache: {exact_cache_path.name}")
                
                if exact_cache_path.exists():
                    try:
                        with open(exact_cache_path, 'r') as f:
                            cache_data = json.load(f)
                        
                        # Check for highlight history
                        if 'highlight_history' in cache_data:
                            history.extend(cache_data['highlight_history'])
                            print(f"  ✓ Found {len(cache_data['highlight_history'])} entries in exact signature cache")
                        
                        # Check for legacy highlight segments
                        elif 'highlight_segments' in cache_data and cache_data['highlight_segments']:
                            segments = cache_data.get('highlight_segments', [])
                            metadata = cache_data.get('highlight_metadata', {})
                            history.append({
                                'segments': segments,
                                'segments_count': len(segments),
                                'total_duration': sum(end - start for start, end in segments),
                                'parameters': metadata.get('parameters', {}),
                                'score_info': metadata.get('score_info', {}),
                                'created_at': metadata.get('created_at', 'Unknown')
                            })
                            print(f"  ✓ Added legacy history from exact signature cache")
                    except Exception as e:
                        print(f"  ⚠️ Error reading exact cache: {e}")
            
            # Method 2: Look for any cache file with this video hash
            print(f"  - Looking for any cache files with hash {video_hash}")
            matching_files = list(cache_dir.glob(f"{video_hash}*.cache.json"))
            print(f"  - Found {len(matching_files)} cache files matching hash:")
            
            for cache_file in matching_files:
                try:
                    print(f"    - Reading {cache_file.name}")
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    # Check for highlight history
                    if 'highlight_history' in cache_data:
                        print(f"      - Found highlight_history with {len(cache_data['highlight_history'])} entries")
                        history.extend(cache_data['highlight_history'])
                    
                    # Check for legacy highlight segments
                    elif 'highlight_segments' in cache_data and cache_data['highlight_segments']:
                        segments = cache_data.get('highlight_segments', [])
                        metadata = cache_data.get('highlight_metadata', {})
                        print(f"      - Found legacy highlight_segments with {len(segments)} segments")
                        history.append({
                            'segments': segments,
                            'segments_count': len(segments),
                            'total_duration': sum(end - start for start, end in segments),
                            'parameters': metadata.get('parameters', {}),
                            'score_info': metadata.get('score_info', {}),
                            'created_at': metadata.get('created_at', 'Unknown')
                        })
                except Exception as e:
                    print(f"      ⚠️ Error reading {cache_file.name}: {e}")
                    continue
            
            # Remove duplicates based on created_at timestamp
            seen = set()
            unique_history = []
            for entry in history:
                created = entry.get('created_at', '')
                if created not in seen:
                    seen.add(created)
                    unique_history.append(entry)
            
            # Sort by created_at (most recent first)
            unique_history.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            print(f"  📤 Returning {len(unique_history)} unique history entries")
            if unique_history:
                print(f"  - First entry created at: {unique_history[0].get('created_at', 'Unknown')}")
                print(f"  - First entry segments count: {unique_history[0].get('segments_count', 0)}")
            
            return unique_history
            
        except Exception as e:
            print(f"  ❌ Exception in get_highlight_history: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        with self._lock:
            analysis_files = list(self.cache_dir.glob("*.cache.json"))
            analysis_count = len(analysis_files)

            highlights_root = self.cache_dir / "highlights"
            highlight_count = 0
            total_versions = 0
            if highlights_root.exists():
                for video_dir in highlights_root.iterdir():
                    if video_dir.is_dir():
                        versions = len(list(video_dir.glob("*.json")))
                        if versions > 0:
                            highlight_count += 1
                            total_versions += versions

            analysis_size = sum(p.stat().st_size for p in analysis_files)
            highlight_size = 0
            if highlights_root.exists():
                highlight_size = sum(p.stat().st_size for p in highlights_root.rglob("*.json"))

            total_hits = self.stats["hits"] + self.stats["highlight_hits"]
            total_misses = self.stats["misses"] + self.stats["highlight_misses"]
            total_requests = total_hits + total_misses

            return {
                "analysis_entries": analysis_count,
                "videos_with_highlights": highlight_count,
                "total_highlight_versions": total_versions,
                "analysis_cache_size_mb": analysis_size / (1024 * 1024),
                "highlight_cache_size_mb": highlight_size / (1024 * 1024),
                "total_cache_size_mb": (analysis_size + highlight_size) / (1024 * 1024),
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "highlight_hits": self.stats["highlight_hits"],
                "highlight_misses": self.stats["highlight_misses"],
                "total_hit_rate": total_hits / max(1, total_requests),
                "analysis_hit_rate": self.stats["hits"] / max(1, (self.stats["hits"] + self.stats["misses"])),
                "highlight_hit_rate": self.stats["highlight_hits"]
                / max(1, (self.stats["highlight_hits"] + self.stats["highlight_misses"])),
            }


__all__ = [
    "VideoAnalysisCache",
    "CachedAnalysisData",
    "HighlightSegment",
    "HighlightMetadata",
]


if __name__ == "__main__":
    # Minimal sanity test that does NOT require a real video file:
    cache = VideoAnalysisCache()
    print("✅ VideoAnalysisCache module loaded (no file operations executed).")