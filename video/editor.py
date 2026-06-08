"""FFmpeg-based Video Editor - Timeline Construction and Clip Assembly"""

import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import json
import tempfile


@dataclass
class TimelineClip:
    """Clip placement on timeline"""
    clip_path: Path
    start_time: float  # Time in output timeline
    duration: float  # Duration to use from this clip
    source_offset: float = 0.0  # Where to start in source clip
    transitions: Optional[Dict] = None  # Transition effects


class VideoEditor:
    """Build video timeline using FFmpeg with clips and effects"""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
    
    def create_concat_file(self, timeline_clips: List[TimelineClip]) -> str:
        """
        Create FFmpeg concat demuxer file for clip sequence.
        
        Args:
            timeline_clips: List of clips with timings
        
        Returns:
            Path to concat file
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for clip in timeline_clips:
                f.write(f"file '{clip.clip_path}'\n")
                f.write(f"duration {clip.duration}\n")
            return f.name
    
    def build_filter_graph(self, timeline_clips: List[TimelineClip], 
                          transitions: Optional[List[Dict]] = None) -> str:
        """
        Build FFmpeg filter graph for all effects and transitions.
        
        Args:
            timeline_clips: Video clips on timeline
            transitions: Transition configurations
        
        Returns:
            Filter graph string
        """
        
        filters = []
        
        # Build clip inputs
        for i, clip in enumerate(timeline_clips):
            if clip.transitions:
                # Add transitions/effects for this clip
                trans = clip.transitions
                
                # Trim clip to duration
                if clip.source_offset > 0 or clip.duration > 0:
                    filters.append(f"[{i}:v]trim=start={clip.source_offset}:duration={clip.duration}[v{i}]")
                else:
                    filters.append(f"[{i}:v]copy[v{i}]")
        
        # Complex concatenation with transitions
        if len(filters) > 0:
            return ";".join(filters)
        
        return ""
    
    def assemble_timeline(self, timeline_clips: List[TimelineClip], 
                         output_path: Path,
                         fps: int = 30,
                         width: int = 1920,
                         height: int = 1080) -> bool:
        """
        Assemble video timeline from clips.
        
        Args:
            timeline_clips: Clips to assemble
            output_path: Output video path
            fps: Output frames per second
            width: Output width
            height: Output height
        
        Returns:
            True if successful
        """
        
        try:
            # Create concat file
            concat_file = self.create_concat_file(timeline_clips)
            
            # Build FFmpeg command
            cmd = [
                self.ffmpeg_path,
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-r', str(fps),
                '-s', f'{width}x{height}',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-y',
                str(output_path)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return False
            
            return True
        
        except Exception as e:
            print(f"Error assembling timeline: {e}")
            return False
    
    def extract_audio(self, video_path: Path, audio_output: Path) -> bool:
        """
        Extract audio stream from video.
        """
        
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', str(video_path),
                '-q:a', '0',
                '-map', 'a',
                '-y',
                str(audio_output)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return False
    
    def replace_audio(self, video_path: Path, audio_path: Path, 
                     output_path: Path) -> bool:
        """
        Replace audio track in video with new audio.
        """
        
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', str(video_path),
                '-i', str(audio_path),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                '-y',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        
        except Exception as e:
            print(f"Error replacing audio: {e}")
            return False
