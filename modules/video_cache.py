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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import threading
import shutil


# ========== DATA CLASSES (unchanged external API) ==========

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

    def get_transcript_segments(self) -> List[Dict[str, Any]]:
        return self.transcript.get("segments", [])

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
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_cache_size = int(max_cache_size_mb) * 1024 * 1024
        self.enable_highlight_cache = bool(enable_highlight_cache)
        self.max_highlight_versions = int(max_highlight_versions)

        self._lock = threading.RLock()

        # enhanced directory structure
        (self.cache_dir / "highlights").mkdir(exist_ok=True)
        (self.cache_dir / "temp").mkdir(exist_ok=True)

        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "highlight_hits": 0,
            "highlight_misses": 0,
        }

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

    def exists(self, video_path: str) -> bool:
        with self._lock:
            return self._get_cache_path(video_path).exists()

    def save(self, video_path: str, analysis_data: Dict[str, Any]) -> None:
        with self._lock:
            cache_path = self._get_cache_path(video_path)

            cache_data = {
                "video_path": str(Path(video_path).absolute()),
                "video_hash": self._get_video_hash(video_path),
                "cached_at": datetime.now().isoformat(),
                "cache_version": "1.0",
                **analysis_data,
            }

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            self.stats["saves"] += 1
            print(f"✓ Cache saved: {cache_path}")

    def load(self, video_path: str) -> Optional[Dict[str, Any]]:
        with self._lock:
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

    def save_highlight_segments(
        self,
        video_path: str,
        parameters: Dict[str, Any],
        segments: List[Tuple[float, float]],
        segments_metadata: Optional[List[Dict[str, Any]]] = None,
        score_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.enable_highlight_cache:
            return False

        with self._lock:
            try:
                video_hash = self._get_video_hash(video_path)
                params_hash = self._get_parameters_hash(parameters)

                highlight_segments: List[HighlightSegment] = []
                for i, (start, end) in enumerate(segments):
                    duration = float(end - start)
                    md = segments_metadata[i] if segments_metadata and i < len(segments_metadata) else {}

                    highlight_segments.append(
                        HighlightSegment(
                            start_time=float(start),
                            end_time=float(end),
                            duration=duration,
                            score=float(md.get("score", 0.0)),
                            signals=md.get("signals", {}) or {},
                            primary_reason=md.get("primary_reason", "") or "",
                        )
                    )

                total_duration = sum(float(e - s) for s, e in segments)
                target_duration = parameters.get("exact_duration") or parameters.get("max_duration", 420)
                exact_duration = parameters.get("exact_duration")

                highlight_meta = HighlightMetadata(
                    video_path=str(Path(video_path).absolute()),
                    video_hash=video_hash,
                    parameters_hash=params_hash,
                    total_duration=float(total_duration),
                    target_duration=float(target_duration),
                    duration_mode="EXACT" if exact_duration else "MAX",
                    segments_count=len(segments),
                    parameters=parameters,
                    score_distribution=score_info or {},
                )

                highlight_data = {
                    "metadata": highlight_meta.to_dict(),
                    "segments": [s.to_dict() for s in highlight_segments],
                    "created_at": datetime.now().isoformat(),
                }

                # prepare paths
                hl_dir = self._highlight_dir(video_hash)
                hl_dir.mkdir(exist_ok=True)

                highlight_path = hl_dir / f"{params_hash}.json"
                temp_path = self.cache_dir / "temp" / f"highlight_{video_hash}_{params_hash}.tmp"

                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(highlight_data, f, indent=2, ensure_ascii=False)

                # atomic move
                shutil.move(str(temp_path), str(highlight_path))

                self._cleanup_old_highlights(video_hash)

                print("✅ Highlight segments cached")
                return True

            except Exception as e:
                print(f"❌ Failed to save highlight cache: {e}")
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

    def get_highlight_history(self, video_path: str) -> List[Dict[str, Any]]:
        with self._lock:
            try:
                video_hash = self._get_video_hash(video_path)
                hl_dir = self._highlight_dir(video_hash)
                if not hl_dir.exists():
                    return []

                history: List[Dict[str, Any]] = []
                for fpath in hl_dir.glob("*.json"):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            highlight_data = json.load(f)

                        metadata = highlight_data.get("metadata", {}) or {}
                        segs = highlight_data.get("segments", []) or []

                        history.append(
                            {
                                "created_at": highlight_data.get("created_at"),
                                "parameters": metadata.get("parameters", {}),
                                "total_duration": metadata.get("total_duration", 0),
                                "segments_count": len(segs),
                                "segments": [(s["start_time"], s["end_time"]) for s in segs],
                            }
                        )
                    except Exception:
                        continue

                history.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                return history

            except Exception as e:
                print(f"⚠️ Error getting highlight history: {e}")
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
