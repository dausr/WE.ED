"""Video Transitions - Blend Effects, Wobble Zoom, Stutter, and More"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TransitionType(Enum):
    """Types of transitions"""
    CROSS_FADE = "cross_fade"
    BLEND = "blend"
    WOBBLE_ZOOM = "wobble_zoom"
    STUTTER = "stutter"
    WIPE = "wipe"
    PIXELATE = "pixelate"
    MOTION_BLUR = "motion_blur"


@dataclass
class TransitionConfig:
    """Configuration for transition effect"""
    transition_type: TransitionType
    duration: float = 0.3
    intensity: float = 1.0
    direction: str = "center_outward"  # For wipes
    frequency: float = 3.0  # For wobble
    color: Tuple[int, int, int] = (0, 0, 0)


class TransitionEffects:
    """Generate various video transition effects"""
    
    def __init__(self, fps: int = 30):
        self.fps = fps
    
    def cross_fade(self, frame1: np.ndarray, frame2: np.ndarray, 
                  progress: float) -> np.ndarray:
        """
        Simple cross-fade between two frames.
        
        Args:
            frame1: First frame (outgoing)
            frame2: Second frame (incoming)
            progress: Transition progress (0 to 1)
        
        Returns:
            Blended frame
        """
        
        alpha = progress
        blended = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
        return blended.astype(np.uint8)
    
    def blend_edges(self, frame1: np.ndarray, frame2: np.ndarray,
                   progress: float, edge_width: int = 30) -> np.ndarray:
        """
        Blend transition with soft edges (more sophisticated).
        """
        
        h, w = frame1.shape[:2]
        
        # Create mask with smooth gradient
        mask = np.zeros((h, w), dtype=np.float32)
        transition_pos = int(progress * w)
        
        for x in range(max(0, transition_pos - edge_width), min(w, transition_pos + edge_width)):
            local_progress = (x - (transition_pos - edge_width)) / (2 * edge_width)
            mask[:, x] = np.clip(local_progress, 0, 1)
        
        # Fill edges
        mask[:, :max(0, transition_pos - edge_width)] = 0
        mask[:, min(w, transition_pos + edge_width):] = 1
        
        # Apply mask
        mask = np.stack([mask] * 3, axis=2)
        blended = (frame1 * (1 - mask) + frame2 * mask).astype(np.uint8)
        
        return blended
    
    def wobble_zoom(self, frame1: np.ndarray, frame2: np.ndarray,
                   progress: float, intensity: float = 0.8,
                   frequency: float = 3.0) -> np.ndarray:
        """
        Bass-intensity wobble zoom from center outward.
        
        Args:
            frame1: Current frame
            frame2: Next frame
            progress: Transition progress (0 to 1)
            intensity: Wobble intensity (0 to 1)
            frequency: Wobble frequency (Hz-like)
        
        Returns:
            Wobbled frame
        """
        
        h, w = frame1.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Calculate wobble
        wobble = np.sin(progress * frequency * np.pi * 2) * intensity
        zoom = 1.0 + (progress * 0.2 + wobble * 0.1)  # Progressive zoom + wobble
        
        # Create transformation matrix
        M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom)
        
        # Apply zoom
        frame1_zoomed = cv2.warpAffine(frame1, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        frame2_zoomed = cv2.warpAffine(frame2, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Cross-fade between zoomed frames
        result = self.cross_fade(frame1_zoomed, frame2_zoomed, progress)
        
        return result
    
    def stutter(self, frame: np.ndarray, progress: float,
               stutter_count: int = 3, stutter_intensity: float = 0.3) -> np.ndarray:
        """
        Stutter effect - rapid frame repeats synchronized to beat.
        
        Args:
            frame: Frame to stutter
            progress: Transition progress (0 to 1)
            stutter_count: Number of stutters
            stutter_intensity: How much to vary (0 to 1)
        
        Returns:
            Stuttered frame
        """
        
        # Create stutter pattern
        stutter_pattern = np.sin(progress * stutter_count * np.pi * 2)
        stutter_effect = np.clip(stutter_pattern * stutter_intensity, -0.5, 0.5)
        
        # Apply stutter as brightness modulation + slight scale
        h, w = frame.shape[:2]
        scale = 1.0 + stutter_effect * 0.05
        
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 0, scale)
        result = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Modulate brightness
        brightness = 1.0 + stutter_effect * 0.1
        result = cv2.convertScaleAbs(result, alpha=brightness, beta=0)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def wipe(self, frame1: np.ndarray, frame2: np.ndarray,
            progress: float, direction: str = "left_to_right") -> np.ndarray:
        """
        Wipe transition (reveal new frame progressively).
        
        Args:
            frame1: Outgoing frame
            frame2: Incoming frame
            progress: Transition progress (0 to 1)
            direction: Direction of wipe ("left_to_right", "top_to_bottom", etc.)
        
        Returns:
            Wiped frame
        """
        
        h, w = frame1.shape[:2]
        result = frame1.copy()
        
        if direction == "left_to_right":
            wipe_pos = int(progress * w)
            result[:, wipe_pos:] = frame2[:, wipe_pos:]
        elif direction == "top_to_bottom":
            wipe_pos = int(progress * h)
            result[wipe_pos:, :] = frame2[wipe_pos:, :]
        elif direction == "center_outward":
            # Wipe from center outward
            max_dist = int(progress * np.sqrt(h**2 + w**2) / 2)
            cy, cx = h // 2, w // 2
            
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            mask = dist < max_dist
            result[mask] = frame2[mask]
        
        return result
    
    def pixelate_transition(self, frame1: np.ndarray, frame2: np.ndarray,
                           progress: float, pixel_size_max: int = 20) -> np.ndarray:
        """
        Pixelate transition effect.
        
        Args:
            frame1: Outgoing frame
            frame2: Incoming frame
            progress: Transition progress (0 to 1)
            pixel_size_max: Maximum pixel block size
        
        Returns:
            Pixelated transition
        """
        
        # Calculate pixel size based on progress
        pixel_size = int(pixel_size_max * (1 - progress))
        
        if pixel_size <= 1:
            return self.cross_fade(frame1, frame2, progress)
        
        # Pixelate frame1
        frame1_small = cv2.resize(frame1, (frame1.shape[1] // pixel_size, frame1.shape[0] // pixel_size))
        frame1_pixelated = cv2.resize(frame1_small, (frame1.shape[1], frame1.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Blend pixelated with frame2
        result = self.cross_fade(frame1_pixelated, frame2, progress)
        
        return result
    
    def motion_blur_transition(self, frame1: np.ndarray, frame2: np.ndarray,
                              progress: float, blur_intensity: int = 15) -> np.ndarray:
        """
        Motion blur during transition.
        """
        
        # Create motion blur kernel
        kernel_size = int(blur_intensity * progress) | 1  # Make odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply blur
        frame1_blurred = cv2.filter2D(frame1, -1, kernel)
        frame2_blurred = cv2.filter2D(frame2, -1, kernel)
        
        # Blend
        result = self.cross_fade(frame1_blurred, frame2_blurred, progress)
        
        return result
    
    def generate_transition(self, frame1: np.ndarray, frame2: np.ndarray,
                           config: TransitionConfig) -> List[np.ndarray]:
        """
        Generate all frames for a transition.
        
        Args:
            frame1: First frame
            frame2: Second frame
            config: Transition configuration
        
        Returns:
            List of transition frames
        """
        
        num_frames = int(config.duration * self.fps)
        frames = []
        
        for i in range(num_frames):
            progress = i / max(1, num_frames - 1)
            
            if config.transition_type == TransitionType.CROSS_FADE:
                frame = self.cross_fade(frame1, frame2, progress)
            elif config.transition_type == TransitionType.BLEND:
                frame = self.blend_edges(frame1, frame2, progress)
            elif config.transition_type == TransitionType.WOBBLE_ZOOM:
                frame = self.wobble_zoom(frame1, frame2, progress, 
                                        config.intensity, config.frequency)
            elif config.transition_type == TransitionType.STUTTER:
                frame = self.stutter(frame1, progress)
            elif config.transition_type == TransitionType.WIPE:
                frame = self.wipe(frame1, frame2, progress, config.direction)
            elif config.transition_type == TransitionType.PIXELATE:
                frame = self.pixelate_transition(frame1, frame2, progress)
            elif config.transition_type == TransitionType.MOTION_BLUR:
                frame = self.motion_blur_transition(frame1, frame2, progress)
            else:
                frame = self.cross_fade(frame1, frame2, progress)
            
            frames.append(frame)
        
        return frames
