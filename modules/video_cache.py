"""
Video Analysis Cache Module
Caches all analysis results (transcript, objects, actions, scenes, audio, motion)
to avoid re-processing when changing highlight parameters.
"""

import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pickle


class VideoAnalysisCache:
    """
    Manages caching of video analysis results to enable fast parameter changes
    without re-processing the entire video.
    """
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_video_hash(self, video_path: str) -> str:
        """
        Generate a unique hash for a video file based on:
        - File path
        - File size
        - Modification time
        
        This is faster than hashing the entire video content.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            SHA256 hash string
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create hash from file metadata
        stat = video_path.stat()
        hash_string = f"{video_path.absolute()}_{stat.st_size}_{stat.st_mtime}"
        
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def _get_cache_path(self, video_path: str) -> Path:
        """
        Get the cache file path for a video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the cache file
        """
        video_hash = self._get_video_hash(video_path)
        return self.cache_dir / f"{video_hash}.cache.json"
    
    def exists(self, video_path: str) -> bool:
        """
        Check if a cache exists for the given video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if cache exists, False otherwise
        """
        cache_path = self._get_cache_path(video_path)
        return cache_path.exists()
    
    def save(self, video_path: str, analysis_data: Dict[str, Any]) -> None:
        """
        Save analysis results to cache.
        
        Args:
            video_path: Path to the video file
            analysis_data: Dictionary containing all analysis results
                Expected keys:
                - transcript: List of transcript segments
                - objects: List of detected objects
                - actions: List of detected actions
                - scenes: List of scene boundaries
                - audio_peaks: List of audio peak data
                - motion_scores: List of motion scores
                - video_metadata: Dict with duration, fps, resolution
        """
        cache_path = self._get_cache_path(video_path)
        
        # Add cache metadata
        cache_data = {
            "video_path": str(Path(video_path).absolute()),
            "video_hash": self._get_video_hash(video_path),
            "cached_at": datetime.now().isoformat(),
            "cache_version": "1.0",
            **analysis_data
        }
        
        # Save as JSON (human-readable)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Cache saved: {cache_path}")
    
    def load(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Load analysis results from cache.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing cached analysis data, or None if not found
        """
        cache_path = self._get_cache_path(video_path)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Verify the cache is for the same video version
            current_hash = self._get_video_hash(video_path)
            if cache_data.get("video_hash") != current_hash:
                print("⚠ Cache is outdated (video file changed), will re-process")
                return None
            
            print(f"✓ Cache loaded from: {cache_path}")
            print(f"  Cached at: {cache_data.get('cached_at', 'unknown')}")
            return cache_data
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠ Cache file corrupted: {e}, will re-process")
            return None
    
    def invalidate(self, video_path: str) -> bool:
        """
        Delete cache for a specific video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            True if cache was deleted, False if not found
        """
        cache_path = self._get_cache_path(video_path)
        
        if cache_path.exists():
            cache_path.unlink()
            print(f"✓ Cache deleted: {cache_path}")
            return True
        return False
    
    def clear_all(self) -> int:
        """
        Clear all cached data.
        
        Returns:
            Number of cache files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.cache.json"):
            cache_file.unlink()
            count += 1
        
        print(f"✓ Cleared {count} cache file(s)")
        return count
    
    def list_cached_videos(self) -> List[Dict[str, Any]]:
        """
        List all cached videos with their metadata.
        
        Returns:
            List of dictionaries containing cache information
        """
        cached_videos = []
        
        for cache_file in self.cache_dir.glob("*.cache.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                cached_videos.append({
                    "video_path": cache_data.get("video_path", "unknown"),
                    "cached_at": cache_data.get("cached_at", "unknown"),
                    "cache_file": str(cache_file),
                    "video_metadata": cache_data.get("video_metadata", {})
                })
            except (json.JSONDecodeError, KeyError):
                continue
        
        return cached_videos
    
    def get_cache_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Get information about cached data without loading the full cache.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with cache metadata, or None if not found
        """
        cache_path = self._get_cache_path(video_path)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Return only metadata
            return {
                "video_path": cache_data.get("video_path"),
                "cached_at": cache_data.get("cached_at"),
                "cache_version": cache_data.get("cache_version"),
                "has_transcript": "transcript" in cache_data,
                "has_objects": "objects" in cache_data,
                "has_actions": "actions" in cache_data,
                "has_scenes": "scenes" in cache_data,
                "has_audio_peaks": "audio_peaks" in cache_data,
                "has_motion_scores": "motion_scores" in cache_data,
                "video_metadata": cache_data.get("video_metadata", {}),
                "cache_size_mb": cache_path.stat().st_size / (1024 * 1024)
            }
        except (json.JSONDecodeError, KeyError):
            return None


class CachedAnalysisData:
    """
    Data class to hold cached analysis results in a structured way.
    """
    
    def __init__(self, cache_data: Dict[str, Any]):
        """
        Initialize from cache data dictionary.
        
        Args:
            cache_data: Dictionary loaded from cache
        """
        self.video_path = cache_data.get("video_path")
        self.video_hash = cache_data.get("video_hash")
        self.cached_at = cache_data.get("cached_at")
        
        # Video metadata
        self.video_metadata = cache_data.get("video_metadata", {})
        self.duration = self.video_metadata.get("duration", 0)
        self.fps = self.video_metadata.get("fps", 30)
        self.resolution = self.video_metadata.get("resolution", "unknown")
        
        # Analysis results
        self.transcript = cache_data.get("transcript", {"segments": []})
        self.objects = cache_data.get("objects", [])
        self.actions = cache_data.get("actions", [])
        self.scenes = cache_data.get("scenes", [])
        self.audio_peaks = cache_data.get("audio_peaks", [])
        self.motion_scores = cache_data.get("motion_scores", [])
    
    def get_transcript_segments(self) -> List[Dict[str, Any]]:
        """Get transcript segments."""
        return self.transcript.get("segments", [])
    
    def get_objects_in_timerange(self, start: float, end: float) -> List[Dict[str, Any]]:
        """
        Get objects detected within a time range.
        
        Args:
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            List of object detections in the time range
        """
        return [
            obj for obj in self.objects
            if start <= obj.get("timestamp", 0) <= end
        ]
    
    def get_actions_in_timerange(self, start: float, end: float) -> List[Dict[str, Any]]:
        """
        Get actions detected within a time range.
        
        Args:
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            List of action detections in the time range
        """
        return [
            action for action in self.actions
            if start <= action.get("timestamp", 0) <= end
        ]
    
    def get_scenes_in_timerange(self, start: float, end: float) -> List[Dict[str, Any]]:
        """
        Get scene boundaries within a time range.
        
        Args:
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            List of scenes in the time range
        """
        return [
            scene for scene in self.scenes
            if not (scene.get("end", 0) < start or scene.get("start", 0) > end)
        ]


# Example usage
if __name__ == "__main__":
    # Initialize cache
    cache = VideoAnalysisCache(cache_dir="./cache")
    
    # Example: Saving analysis results
    example_video = "video.mp4"
    analysis_results = {
        "video_metadata": {
            "duration": 3600,
            "fps": 30,
            "resolution": "1920x1080"
        },
        "transcript": {
            "segments": [
                {"start": 0.0, "end": 5.2, "text": "Hello world", "language": "en"}
            ]
        },
        "objects": [
            {"timestamp": 1.5, "frame": 45, "class": "person", "confidence": 0.95}
        ],
        "actions": [
            {"timestamp": 2.3, "action": "jumping", "confidence": 0.87}
        ],
        "scenes": [
            {"start": 0.0, "end": 10.5},
            {"start": 10.5, "end": 25.3}
        ],
        "audio_peaks": [
            {"timestamp": 5.2, "amplitude": 0.89}
        ],
        "motion_scores": [
            {"timestamp": 1.0, "score": 0.65}
        ]
    }
    
    # Save to cache
    # cache.save(example_video, analysis_results)
    
    # Load from cache
    # cached_data = cache.load(example_video)
    # if cached_data:
    #     analysis = CachedAnalysisData(cached_data)
    #     print(f"Duration: {analysis.duration}s")
    #     print(f"FPS: {analysis.fps}")
    
    # List all cached videos
    cached_videos = cache.list_cached_videos()
    print(f"Found {len(cached_videos)} cached video(s)")