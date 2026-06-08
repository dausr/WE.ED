"""Semantic Analysis Module - AI Embeddings and Song Context Analysis"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

try:
    import torch
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


@dataclass
class SongContext:
    """Semantic context extracted from song metadata and audio"""
    title: str
    artist: str
    genre: str
    bpm: float
    duration: float
    mood: str  # energetic, calm, melancholic, happy, etc.
    energy_level: float  # 0-1
    danceability: float  # 0-1
    mood_embedding: Optional[np.ndarray] = None
    audio_embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'title': self.title,
            'artist': self.artist,
            'genre': self.genre,
            'bpm': self.bpm,
            'duration': self.duration,
            'mood': self.mood,
            'energy_level': self.energy_level,
            'danceability': self.danceability
        }


class SongContextAnalyzer:
    """Analyze song metadata to extract semantic context"""
    
    def __init__(self):
        self.mood_keywords = {
            'energetic': ['electronic', 'dance', 'edm', 'upbeat', 'fast', 'vibrant'],
            'calm': ['ambient', 'chill', 'relaxing', 'mellow', 'slow', 'peaceful'],
            'melancholic': ['sad', 'dark', 'blues', 'ballad', 'emotional', 'minor'],
            'happy': ['pop', 'uplifting', 'cheerful', 'sunny', 'bright', 'major'],
            'aggressive': ['metal', 'punk', 'rock', 'hard', 'heavy', 'intense'],
            'groovy': ['funk', 'soul', 'hip-hop', 'groove', 'rhythmic', 'syncopated']
        }
    
    def analyze(self, metadata: Dict, audio_analysis: Dict) -> SongContext:
        """
        Analyze song to extract semantic context.
        
        Args:
            metadata: Song metadata (title, artist, genre, etc.)
            audio_analysis: Audio analysis results (bpm, beats, etc.)
        """
        
        # Extract metadata
        title = metadata.get('title', 'Unknown')
        artist = metadata.get('artist', 'Unknown')
        genre = metadata.get('genre', 'Unknown').lower()
        duration = metadata.get('duration', 0)
        bpm = audio_analysis.get('bpm', 120)
        
        # Detect mood from genre and metadata
        mood = self._detect_mood(genre, bpm)
        
        # Compute energy and danceability
        energy_level = self._compute_energy(bpm, audio_analysis)
        danceability = self._compute_danceability(bpm, audio_analysis)
        
        return SongContext(
            title=title,
            artist=artist,
            genre=genre,
            bpm=float(bpm),
            duration=float(duration),
            mood=mood,
            energy_level=float(energy_level),
            danceability=float(danceability)
        )
    
    def _detect_mood(self, genre: str, bpm: float) -> str:
        """Detect mood from genre and BPM"""
        
        genre_lower = genre.lower()
        
        # Score each mood
        mood_scores = {mood: 0 for mood in self.mood_keywords}
        
        # Check genre keywords
        for mood, keywords in self.mood_keywords.items():
            for keyword in keywords:
                if keyword in genre_lower:
                    mood_scores[mood] += 2
        
        # Check BPM heuristics
        if bpm > 130:
            mood_scores['energetic'] += 2
            mood_scores['aggressive'] += 1
        elif bpm < 100:
            mood_scores['calm'] += 2
            mood_scores['melancholic'] += 1
        else:
            mood_scores['groovy'] += 1
        
        # Return highest scoring mood
        return max(mood_scores, key=mood_scores.get)
    
    def _compute_energy(self, bpm: float, audio_analysis: Dict) -> float:
        """
        Compute energy level (0-1) from audio features.
        """
        
        # BPM contribution (normalized)
        bpm_energy = min(bpm / 180, 1.0)
        
        # Onset strength (if available)
        onset_energy = audio_analysis.get('avg_onset_strength', 0.5)
        
        # Combined energy
        energy = 0.6 * bpm_energy + 0.4 * onset_energy
        return float(min(energy, 1.0))
    
    def _compute_danceability(self, bpm: float, audio_analysis: Dict) -> float:
        """
        Compute danceability (0-1) - how suitable for dancing.
        """
        
        # BPM sweet spot for dancing: 100-130
        bpm_score = 1.0 - abs(bpm - 115) / 120  # Peak at 115 BPM
        bpm_score = max(0, min(bpm_score, 1.0))
        
        # Rhythm regularity (if available)
        rhythm_score = audio_analysis.get('rhythm_regularity', 0.5)
        
        danceability = 0.7 * bpm_score + 0.3 * rhythm_score
        return float(min(danceability, 1.0))


class SemanticEmbeddingModel:
    """Generate semantic embeddings for clips and songs using CLIP or similar models"""
    
    def __init__(self, model_name: str = "clip-vit-base-patch32", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.preprocessor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize CLIP or fallback embedding model"""
        
        if CLIP_AVAILABLE and "clip" in self.model_name.lower():
            try:
                self.model, self.preprocessor = clip.load(self.model_name, device=self.device)
                self.embedding_dim = self.model.embed_dim
            except Exception as e:
                print(f"Failed to load CLIP: {e}, using fallback")
                self._initialize_fallback()
        else:
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback embedding model if CLIP unavailable"""
        self.embedding_dim = 512
        self.model = None
        print("Using random embedding fallback (install transformers for real embeddings)")
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for text descriptions.
        
        Args:
            texts: List of text descriptions
        
        Returns:
            Array of embeddings (shape: [len(texts), embedding_dim])
        """
        
        if not texts:
            return np.array([])
        
        if self.model is None:
            # Fallback: random embeddings (for testing)
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
        
        try:
            text_tokens = clip.tokenize(texts).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
            return text_features.cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"Error encoding text: {e}")
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
    
    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate embedding for single video frame.
        
        Args:
            frame: Image array (BGR, 0-255)
        
        Returns:
            Embedding vector
        """
        
        if self.model is None:
            return np.random.randn(self.embedding_dim).astype(np.float32)
        
        try:
            from PIL import Image
            import cv2
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Preprocess
            image_tensor = self.preprocessor(pil_image).unsqueeze(0).to(self.device)
            
            # Encode
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
            
            return image_features.squeeze(0).cpu().numpy().astype(np.float32)
        
        except Exception as e:
            print(f"Error encoding frame: {e}")
            return np.random.randn(self.embedding_dim).astype(np.float32)
    
    def encode_frames_batch(self, frames: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple frames efficiently.
        
        Args:
            frames: List of image arrays
            batch_size: Processing batch size
        
        Returns:
            Array of embeddings
        """
        
        if self.model is None:
            return np.random.randn(len(frames), self.embedding_dim).astype(np.float32)
        
        embeddings = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            
            try:
                from PIL import Image
                import cv2
                
                # Convert all frames
                pil_images = []
                for frame in batch_frames:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_images.append(Image.fromarray(rgb_frame))
                
                # Batch preprocess
                image_tensors = torch.stack([
                    self.preprocessor(img) for img in pil_images
                ]).to(self.device)
                
                # Batch encode
                with torch.no_grad():
                    batch_embeddings = self.model.encode_image(image_tensors)
                
                embeddings.append(batch_embeddings.cpu().numpy())
            
            except Exception as e:
                print(f"Error in batch encoding: {e}")
                embeddings.append(np.random.randn(len(batch_frames), self.embedding_dim))
        
        return np.vstack(embeddings).astype(np.float32)


class SemanticSegmenter:
    """Segment song into semantic sections with embeddings"""
    
    def __init__(self, embedding_model: SemanticEmbeddingModel):
        self.embedding_model = embedding_model
    
    def segment_by_beats(self, beats: np.ndarray, context: SongContext) -> List[Dict]:
        """
        Create semantic segments aligned to beats.
        
        Args:
            beats: Beat times in seconds
            context: Song context from analyzer
        
        Returns:
            List of segments with embeddings
        """
        
        segments = []
        
        # Create mood-based text prompts for each segment
        mood_description = self._get_mood_description(context)
        
        # Generate mood embedding once
        mood_embedding = self.embedding_model.encode_text([mood_description])[0]
        
        # Create segments at each beat (or grouped)
        group_size = max(1, len(beats) // 8)  # Group beats into ~8 segments
        
        for i in range(0, len(beats), group_size):
            segment_beats = beats[i:i+group_size]
            start_time = float(segment_beats[0])
            end_time = float(segment_beats[-1]) if len(segment_beats) > 0 else start_time
            
            segment = {
                'start': start_time,
                'end': end_time,
                'beat_indices': list(range(i, min(i+group_size, len(beats)))),
                'embedding': mood_embedding,
                'description': mood_description,
                'energy': context.energy_level
            }
            segments.append(segment)
        
        return segments
    
    def _get_mood_description(self, context: SongContext) -> str:
        """
        Generate text description for CLIP embedding based on mood.
        """
        
        energy_desc = "energetic" if context.energy_level > 0.7 else "calm" if context.energy_level < 0.3 else "moderate"
        dance_desc = "danceable" if context.danceability > 0.6 else "not very danceable"
        
        return f"A {energy_desc} {context.mood} music video with {context.genre} music, {dance_desc}, BPM {context.bpm:.0f}"
