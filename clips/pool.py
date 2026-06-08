"""Clip Pool Management - Load, Index, and Manage Video Clips"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import json
from datetime import datetime
import cv2
import numpy as np


@dataclass
class ClipMetadata:
    """Metadata for a single video clip"""
    path: Path
    filename: str
    duration: float
    width: int
    height: int
    fps: float
    total_frames: int
    codec: str
    file_size_mb: float
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    embedding: Optional[np.ndarray] = None  # Semantic embedding
    tags: Dict[str, any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'path': str(self.path),
            'filename': self.filename,
            'duration': self.duration,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'codec': self.codec,
            'file_size_mb': self.file_size_mb,
            'created_at': self.created_at,
            'tags': self.tags
        }


class ClipPool:
    """Manages collection of video clips with indexing and metadata"""
    
    def __init__(self, clips_dir: Path, cache_file: Optional[Path] = None):
        self.clips_dir = Path(clips_dir)
        self.cache_file = cache_file or self.clips_dir / ".clip_cache.json"
        self.clips: List[ClipMetadata] = []
        self.index: Dict[str, ClipMetadata] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
    def load_clips(self, use_cache: bool = True) -> List[ClipMetadata]:
        """Load all video clips from directory"""
        
        # Try loading from cache first
        if use_cache and self.cache_file.exists():
            try:
                return self._load_from_cache()
            except Exception as e:
                print(f"Cache load failed: {e}, rescanning...")
        
        # Scan directory for video files
        video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.m4v'}
        clips = []
        
        for file_path in sorted(self.clips_dir.iterdir()):
            if file_path.suffix.lower() in video_extensions:
                try:
                    metadata = self._extract_metadata(file_path)
                    clips.append(metadata)
                    self.index[metadata.filename] = metadata
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        self.clips = clips
        
        # Save cache
        if use_cache:
            self._save_to_cache()
        
        return self.clips
    
    def _extract_metadata(self, file_path: Path) -> ClipMetadata:
        """Extract video metadata using OpenCV"""
        
        cap = cv2.VideoCapture(str(file_path))
        
        try:
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Get codec
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            # Calculate duration
            duration = total_frames / fps if fps > 0 else 0
            
            # File size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            metadata = ClipMetadata(
                path=file_path,
                filename=file_path.name,
                duration=duration,
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                codec=codec,
                file_size_mb=file_size_mb
            )
            
            return metadata
        
        finally:
            cap.release()
    
    def _save_to_cache(self) -> None:
        """Save clip metadata to cache file"""
        try:
            cache_data = {
                'clips': [clip.to_dict() for clip in self.clips],
                'cached_at': datetime.now().isoformat()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _load_from_cache(self) -> List[ClipMetadata]:
        """Load clip metadata from cache file"""
        with open(self.cache_file, 'r') as f:
            cache_data = json.load(f)
        
        clips = []
        for clip_dict in cache_data['clips']:
            clip = ClipMetadata(
                path=Path(clip_dict['path']),
                filename=clip_dict['filename'],
                duration=clip_dict['duration'],
                width=clip_dict['width'],
                height=clip_dict['height'],
                fps=clip_dict['fps'],
                total_frames=clip_dict['total_frames'],
                codec=clip_dict['codec'],
                file_size_mb=clip_dict['file_size_mb'],
                created_at=clip_dict.get('created_at', datetime.now().timestamp()),
                tags=clip_dict.get('tags', {})
            )
            clips.append(clip)
            self.index[clip.filename] = clip
        
        self.clips = clips
        return self.clips
    
    def get_by_duration(self, min_duration: float = 0, max_duration: float = float('inf')) -> List[ClipMetadata]:
        """Filter clips by duration range"""
        return [
            clip for clip in self.clips 
            if min_duration <= clip.duration <= max_duration
        ]
    
    def get_by_resolution(self, min_width: int = 0, min_height: int = 0) -> List[ClipMetadata]:
        """Filter clips by minimum resolution"""
        return [
            clip for clip in self.clips
            if clip.width >= min_width and clip.height >= min_height
        ]
    
    def get_16_9_clips(self) -> List[ClipMetadata]:
        """Get clips with 16:9 aspect ratio (with tolerance)"""
        tolerance = 0.05
        target_ratio = 16 / 9
        
        result = []
        for clip in self.clips:
            ratio = clip.width / clip.height if clip.height > 0 else 0
            if abs(ratio - target_ratio) < tolerance:
                result.append(clip)
        
        return result
    
    def set_tag(self, clip_filename: str, tag_key: str, tag_value: any) -> None:
        """Add tag to a clip"""
        if clip_filename in self.index:
            self.index[clip_filename].tags[tag_key] = tag_value
    
    def get_by_tag(self, tag_key: str, tag_value: Optional[any] = None) -> List[ClipMetadata]:
        """Find clips by tag"""
        result = []
        for clip in self.clips:
            if tag_key in clip.tags:
                if tag_value is None or clip.tags[tag_key] == tag_value:
                    result.append(clip)
        return result
    
    def set_embedding(self, clip_filename: str, embedding: np.ndarray) -> None:
        """Store semantic embedding for clip"""
        if clip_filename in self.index:
            self.index[clip_filename].embedding = embedding
            self.embeddings_cache[clip_filename] = embedding
    
    def get_embedding(self, clip_filename: str) -> Optional[np.ndarray]:
        """Retrieve semantic embedding for clip"""
        return self.embeddings_cache.get(clip_filename)
    
    def stats(self) -> Dict:
        """Get pool statistics"""
        if not self.clips:
            return {}
        
        durations = [c.duration for c in self.clips]
        sizes = [c.file_size_mb for c in self.clips]
        
        return {
            'total_clips': len(self.clips),
            'total_duration': sum(durations),
            'total_size_gb': sum(sizes) / 1024,
            'avg_duration': np.mean(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            '16_9_clips': len(self.get_16_9_clips()),
        }
    
    def __len__(self) -> int:
        return len(self.clips)
    
    def __getitem__(self, index: int) -> ClipMetadata:
        return self.clips[index]
    
    def __iter__(self):
        return iter(self.clips)
