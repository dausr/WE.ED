"""Video Rendering Pipeline - Final Output Generation"""

import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import tempfile
import json


@dataclass
class RenderConfig:
    """Configuration for video rendering"""
    output_path: Path
    width: int = 1920
    height: int = 1080
    fps: int = 30
    bitrate: str = "8000k"
    codec: str = "libx264"
    preset: str = "medium"  # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
    crf: int = 23  # Quality (0-51, lower is better)
    audio_bitrate: str = "128k"
    format: str = "mp4"


class VideoRenderer:
    """Render final video output with audio synchronization"""
    
    def __init__(self, ffmpeg_path: str = "ffmpeg", ffprobe_path: str = "ffprobe"):
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
    
    def render_from_frames(self, frame_list: List[np.ndarray],
                          audio_path: Optional[Path],
                          config: RenderConfig) -> bool:
        """
        Render video from frame sequence with optional audio.
        
        Args:
            frame_list: List of video frames (BGR, uint8)
            audio_path: Optional path to audio file
            config: Render configuration
        
        Returns:
            True if successful
        """
        
        try:
            import cv2
            
            # Create temporary video file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp_video = Path(tmp.name)
            
            # Write frames using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(tmp_video), fourcc, config.fps, 
                                 (config.width, config.height))
            
            for frame in frame_list:
                # Ensure correct size
                if frame.shape[:2] != (config.height, config.width):
                    frame = cv2.resize(frame, (config.width, config.height))
                out.write(frame)
            
            out.release()
            
            # Add audio if provided
            if audio_path and audio_path.exists():
                return self._add_audio(tmp_video, audio_path, config)
            else:
                # Just copy to output
                import shutil
                shutil.copy(tmp_video, config.output_path)
                tmp_video.unlink()
                return True
        
        except Exception as e:
            print(f"Error rendering frames: {e}")
            return False
    
    def _add_audio(self, video_path: Path, audio_path: Path, 
                  config: RenderConfig) -> bool:
        """
        Add audio track to video.
        """
        
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', str(video_path),
                '-i', str(audio_path),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', config.audio_bitrate,
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                '-y',
                str(config.output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Clean up temp video
            video_path.unlink(missing_ok=True)
            
            return result.returncode == 0
        
        except Exception as e:
            print(f"Error adding audio: {e}")
            return False
    
    def render_from_clips(self, timeline: List[Dict], 
                         audio_path: Optional[Path],
                         config: RenderConfig) -> bool:
        """
        Render video from timeline of clips with transitions.
        
        Args:
            timeline: List of clip placements with transitions
            audio_path: Optional audio file
            config: Render configuration
        
        Returns:
            True if successful
        """
        
        try:
            # Create FFmpeg input file list
            input_list = self._create_input_list(timeline)
            
            # Build filter graph
            filter_graph = self._build_filter_graph(timeline, config)
            
            # Build FFmpeg command
            cmd = [self.ffmpeg_path]
            
            # Add inputs
            for clip_path in self._extract_clip_paths(timeline):
                cmd.extend(['-i', str(clip_path)])
            
            # Add filter graph if any
            if filter_graph:
                cmd.extend(['-filter_complex', filter_graph])
            
            # Add audio
            if audio_path:
                cmd.extend(['-i', str(audio_path)])
                cmd.extend(['-map', '[v]', '-map', str(len(self._extract_clip_paths(timeline))) + ':a'])
            else:
                cmd.extend(['-map', '[v]'])
            
            # Output options
            cmd.extend([
                '-c:v', config.codec,
                '-preset', config.preset,
                '-crf', str(config.crf),
                '-b:v', config.bitrate,
                '-c:a', 'aac',
                '-b:a', config.audio_bitrate,
                '-r', str(config.fps),
                '-s', f'{config.width}x{config.height}',
                '-y',
                str(config.output_path)
            ])
            
            print(f"Rendering: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFmpeg error: {result.stderr}")
                return False
            
            return True
        
        except Exception as e:
            print(f"Error rendering from clips: {e}")
            return False
    
    def _create_input_list(self, timeline: List[Dict]) -> List[Path]:
        """
        Extract unique clip paths from timeline.
        """
        
        clips = []
        seen = set()
        
        for item in timeline:
            clip_path = Path(item['clip_path'])
            if str(clip_path) not in seen:
                clips.append(clip_path)
                seen.add(str(clip_path))
        
        return clips
    
    def _extract_clip_paths(self, timeline: List[Dict]) -> List[Path]:
        """
        Extract clip paths from timeline.
        """
        
        return self._create_input_list(timeline)
    
    def _build_filter_graph(self, timeline: List[Dict], config: RenderConfig) -> str:
        """
        Build FFmpeg complex filter graph for timeline assembly.
        """
        
        filters = []
        
        # Build filter components for each clip
        for i, item in enumerate(timeline):
            duration = item.get('duration', 1.0)
            
            # Trim and scale each clip
            filters.append(
                f"[{i}:v]trim=0:{duration},scale={config.width}:{config.height}[v{i}]"
            )
        
        # Concatenate all clips
        concat_inputs = "".join([f"[v{i}]" for i in range(len(timeline))])
        filters.append(f"{concat_inputs}concat=n={len(timeline)}:v=1:a=0[v]")
        
        return ";".join(filters) if filters else ""
    
    def get_video_info(self, video_path: Path) -> Dict:
        """
        Get video information using ffprobe.
        """
        
        try:
            cmd = [
                self.ffprobe_path,
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            info = json.loads(result.stdout)
            
            return info
        
        except Exception as e:
            print(f"Error getting video info: {e}")
            return {}
    
    def validate_output(self, video_path: Path) -> bool:
        """
        Validate that output video was created successfully.
        """
        
        if not video_path.exists():
            return False
        
        info = self.get_video_info(video_path)
        return 'streams' in info and len(info['streams']) > 0
