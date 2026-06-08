"""Audio Analysis Module - BPM, Beat Detection, Genre from MP3"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum


class AudioGenre(Enum):
    """Audio genre classification"""
    POP = "pop"
    HIP_HOP = "hip_hop"
    ELECTRONIC = "electronic"
    ROCK = "rock"
    INDIE = "indie"
    AMBIENT = "ambient"
    UNKNOWN = "unknown"


@dataclass
class AudioAnalysis:
    """Result of audio analysis"""
    bpm: float
    beats: np.ndarray  # Beat times in seconds
    beat_frames: np.ndarray  # Beat times in frames
    tempo: np.ndarray  # Tempo array
    onset_strength: np.ndarray  # Onset strength envelope
    spectral_centroid: np.ndarray  # Frequency distribution
    mfcc: np.ndarray  # Mel-frequency cepstral coefficients
    genre_estimate: AudioGenre
    duration: float
    sample_rate: int


class AudioAnalyzer:
    """Analyze MP3 audio files for beat-sync video creation"""

    def __init__(self, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def analyze(self, audio_path: str) -> AudioAnalysis:
        """Complete audio analysis pipeline"""
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Beat detection
        bpm, beats_frames = librosa.beat.beat_track(y=y, sr=sr)
        beats = librosa.frames_to_time(beats_frames, sr=sr)
        
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        
        # Tempo
        tempo = librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)[0]
        
        # Spectral features
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        
        # MFCC for timbre analysis
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=self.hop_length)
        
        # Genre estimation (simple heuristic)
        genre = self._estimate_genre(bpm, mfcc, spec_centroid)
        
        return AudioAnalysis(
            bpm=float(bpm),
            beats=beats,
            beat_frames=beats_frames,
            tempo=tempo,
            onset_strength=onset_env,
            spectral_centroid=spec_centroid,
            mfcc=mfcc,
            genre_estimate=genre,
            duration=float(duration),
            sample_rate=sr
        )

    def _estimate_genre(self, bpm: float, mfcc: np.ndarray, spec_centroid: np.ndarray) -> AudioGenre:
        """Simple genre estimation based on audio features"""
        
        mean_centroid = np.mean(spec_centroid)
        mfcc_std = np.std(mfcc)
        
        # Heuristic rules
        if bpm > 120 and mfcc_std > 0.5:
            return AudioGenre.ELECTRONIC
        elif bpm < 100 and mean_centroid < 4000:
            return AudioGenre.AMBIENT
        elif bpm > 100 and bpm < 130:
            return AudioGenre.POP
        elif bpm > 85 and bpm < 115 and mean_centroid > 3000:
            return AudioGenre.HIP_HOP
        else:
            return AudioGenre.UNKNOWN

    def get_beat_positions(self, audio_analysis: AudioAnalysis) -> List[Tuple[float, float]]:
        """Get (start_time, end_time) tuples for each beat"""
        beats = audio_analysis.beats
        beat_durations = np.diff(beats)
        avg_beat_duration = np.mean(beat_durations) if len(beat_durations) > 0 else 0.5
        
        beat_positions = []
        for i, beat_time in enumerate(beats):
            if i < len(beats) - 1:
                end_time = beats[i + 1]
            else:
                end_time = beat_time + avg_beat_duration
            beat_positions.append((float(beat_time), float(end_time)))
        
        return beat_positions
