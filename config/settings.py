"""Global Settings & Configuration Management for WE.ED Framework"""

import os
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseSettings, Field


class AudioSettings(BaseSettings):
    """Audio processing configuration"""
    sample_rate: int = 44100
    n_fft: int = 2048
    hop_length: int = 512
    min_bpm: int = 80
    max_bpm: int = 180


class VideoSettings(BaseSettings):
    """Video output configuration"""
    resolution: str = "1920x1080"
    frame_rate: int = 30
    codec: str = "libx264"
    preset: str = "medium"
    bitrate: str = "8000k"


class ClipSelectionSettings(BaseSettings):
    """Clip selection strategy"""
    avoid_repeats: bool = True
    max_repeat_distance: int = 5
    semantic_threshold: float = 0.7
    diversity_factor: float = 0.3


class TransitionSettings(BaseSettings):
    """Transition effects configuration"""
    blend_fade: bool = True
    fade_duration: float = 0.3
    wobble_intensity: float = 0.8
    wobble_frequency: float = 3.0
    zoom_direction: str = "center_outward"
    stutter_effect: bool = True
    stutter_frames: int = 3


class OutputSettings(BaseSettings):
    """Output file management"""
    version_format: str = "{original_name}_v{version:03d}"
    auto_versioning: bool = True
    preserve_aspect_ratio: bool = True
    no_black_bars: bool = True
    output_format: str = "mp4"


class SemanticSettings(BaseSettings):
    """AI/ML semantic analysis settings"""
    embedding_model: str = "clip-vit-base-patch32"
    device: str = "cuda"  # or "cpu"
    batch_size: int = 32


class WEEDConfig(BaseSettings):
    """Master WE.ED Framework Configuration"""
    audio: AudioSettings = Field(default_factory=AudioSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    clip_selection: ClipSelectionSettings = Field(default_factory=ClipSelectionSettings)
    transitions: TransitionSettings = Field(default_factory=TransitionSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    semantics: SemanticSettings = Field(default_factory=SemanticSettings)

    # Paths
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    sounds_dir: Path = Field(default_factory=lambda: Path.cwd() / "sounds")
    clips_dir: Path = Field(default_factory=lambda: Path.cwd() / "clips")
    output_dir: Path = Field(default_factory=lambda: Path.cwd() / "output")

    @classmethod
    def from_yaml(cls, config_path: str) -> "WEEDConfig":
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file"""
        with open(output_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)


# Global configuration instance
config: Optional[WEEDConfig] = None


def get_config() -> WEEDConfig:
    """Get or initialize global configuration"""
    global config
    if config is None:
        config = WEEDConfig()
    return config


def set_config(new_config: WEEDConfig) -> None:
    """Set global configuration"""
    global config
    config = new_config
