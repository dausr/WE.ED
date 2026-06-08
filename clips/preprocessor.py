"""Clip Preprocessor - Normalization, Frame Extraction, and Metadata"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class ClipFrame:
    """Single frame from a video clip"""
    frame: np.ndarray  # BGR image
    frame_idx: int
    timestamp: float
    
    def to_rgb(self) -> np.ndarray:
        """Convert BGR to RGB"""
        return cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
    
    def resize(self, width: int, height: int) -> 'ClipFrame':
        """Resize frame"""
        resized = cv2.resize(self.frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        return ClipFrame(resized, self.frame_idx, self.timestamp)
    
    def normalize(self) -> np.ndarray:
        """Normalize frame to [0, 1] float32"""
        return self.frame.astype(np.float32) / 255.0


class ClipPreprocessor:
    """Preprocess video clips - normalize, extract frames, compute features"""
    
    def __init__(self, target_width: int = 1920, target_height: int = 1080, preserve_aspect: bool = True):
        self.target_width = target_width
        self.target_height = target_height
        self.preserve_aspect = preserve_aspect
    
    def extract_keyframes(self, clip_path: Path, num_frames: int = 10) -> List[ClipFrame]:
        """Extract evenly-spaced keyframes from clip"""
        
        cap = cv2.VideoCapture(str(clip_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        keyframes = []
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for target_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            
            if ret:
                timestamp = target_idx / fps if fps > 0 else 0
                keyframe = ClipFrame(
                    frame=frame,
                    frame_idx=target_idx,
                    timestamp=timestamp
                )
                keyframes.append(keyframe)
        
        cap.release()
        return keyframes
    
    def get_frame_at(self, clip_path: Path, timestamp: float) -> Optional[ClipFrame]:
        """Extract single frame at specific timestamp"""
        
        cap = cv2.VideoCapture(str(clip_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_idx = int(timestamp * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return ClipFrame(frame, frame_idx, timestamp)
        return None
    
    def normalize_resolution(self, frame: ClipFrame) -> ClipFrame:
        """Resize frame to target resolution"""
        
        if self.preserve_aspect:
            # Calculate scaling to fit within target
            scale = min(
                self.target_width / frame.frame.shape[1],
                self.target_height / frame.frame.shape[0]
            )
            new_width = int(frame.frame.shape[1] * scale)
            new_height = int(frame.frame.shape[0] * scale)
            
            # Resize
            resized = cv2.resize(frame.frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Pad to target size (centered, black padding)
            padded = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
            y_offset = (self.target_height - new_height) // 2
            x_offset = (self.target_width - new_width) // 2
            padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            return ClipFrame(padded, frame.frame_idx, frame.timestamp)
        else:
            # Direct resize (may distort)
            resized = cv2.resize(frame.frame, (self.target_width, self.target_height), 
                               interpolation=cv2.INTER_LANCZOS4)
            return ClipFrame(resized, frame.frame_idx, frame.timestamp)
    
    def detect_black_frames(self, clip_path: Path, threshold: float = 20) -> List[int]:
        """Detect frames that are mostly black (bad frames to avoid)"""
        
        cap = cv2.VideoCapture(str(clip_path))
        black_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale and check brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < threshold:
                black_frames.append(frame_idx)
            
            frame_idx += 1
        
        cap.release()
        return black_frames
    
    def compute_motion_score(self, clip_path: Path, sample_frames: int = 30) -> float:
        """Compute average motion/activity in clip (0-1 scale)"""
        
        cap = cv2.VideoCapture(str(clip_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        motion_scores = []
        frame_indices = np.linspace(0, total_frames - 2, sample_frames, dtype=int)
        
        prev_gray = None
        
        for target_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Resize for faster computation
            small_frame = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Compute optical flow
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_score = np.mean(magnitude) / 10.0  # Normalize
                motion_scores.append(min(motion_score, 1.0))
            
            prev_gray = gray
        
        cap.release()
        
        return float(np.mean(motion_scores)) if motion_scores else 0.0
    
    def extract_color_histogram(self, frame: ClipFrame, bins: int = 32) -> np.ndarray:
        """Extract RGB color histogram from frame"""
        
        # Separate channels
        b, g, r = cv2.split(frame.frame)
        
        # Compute histograms
        hist_r = cv2.calcHist([r], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([g], [0], None, [bins], [0, 256])
        hist_b = cv2.calcHist([b], [0], None, [bins], [0, 256])
        
        # Concatenate and normalize
        histogram = np.concatenate([hist_r, hist_g, hist_b]).flatten()
        histogram = histogram / (np.sum(histogram) + 1e-8)
        
        return histogram
    
    def compute_scene_features(self, clip_path: Path) -> dict:
        """Compute various features of the clip"""
        
        cap = cv2.VideoCapture(str(clip_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Sample frames
        brightness_scores = []
        contrast_scores = []
        saturations = []
        
        sample_size = min(30, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, sample_size, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert to HSV for saturation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            saturation = np.mean(hsv[:, :, 1]) / 255.0
            saturations.append(saturation)
            
            # Brightness and contrast
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 255.0
            
            brightness_scores.append(brightness)
            contrast_scores.append(contrast)
        
        cap.release()
        
        return {
            'avg_brightness': float(np.mean(brightness_scores)) if brightness_scores else 0.5,
            'avg_contrast': float(np.mean(contrast_scores)) if contrast_scores else 0.5,
            'avg_saturation': float(np.mean(saturations)) if saturations else 0.5,
            'motion_score': self.compute_motion_score(clip_path),
            'duration': total_frames / fps if fps > 0 else 0,
            'resolution': {'width': width, 'height': height},
            'fps': fps
        }
