"""Semantic Embeddings Generator - Create embeddings from video clips"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import cv2
from tqdm import tqdm


class EmbeddingGenerator:
    """Generate semantic embeddings for clips from keyframes"""
    
    def __init__(self, embedding_model, num_keyframes: int = 5):
        self.embedding_model = embedding_model
        self.num_keyframes = num_keyframes
    
    def generate_clip_embedding(self, clip_path: Path) -> np.ndarray:
        """
        Generate single embedding for entire clip by averaging keyframe embeddings.
        
        Args:
            clip_path: Path to video clip
        
        Returns:
            Averaged embedding vector
        """
        
        cap = cv2.VideoCapture(str(clip_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return np.zeros(self.embedding_model.embedding_dim, dtype=np.float32)
        
        # Extract keyframes
        keyframe_indices = np.linspace(0, total_frames - 1, self.num_keyframes, dtype=int)
        keyframes = []
        
        for idx in keyframe_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                keyframes.append(frame)
        
        cap.release()
        
        if not keyframes:
            return np.zeros(self.embedding_model.embedding_dim, dtype=np.float32)
        
        # Generate embeddings and average
        embeddings = self.embedding_model.encode_frames_batch(keyframes)
        averaged_embedding = np.mean(embeddings, axis=0)
        
        return averaged_embedding.astype(np.float32)
    
    def generate_batch_embeddings(self, clip_paths: List[Path], verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for multiple clips.
        
        Args:
            clip_paths: List of clip paths
            verbose: Show progress bar
        
        Returns:
            Dictionary mapping clip filename to embedding
        """
        
        embeddings = {}
        iterator = tqdm(clip_paths, desc="Generating embeddings") if verbose else clip_paths
        
        for clip_path in iterator:
            try:
                embedding = self.generate_clip_embedding(clip_path)
                embeddings[clip_path.name] = embedding
            except Exception as e:
                print(f"Error processing {clip_path}: {e}")
                embeddings[clip_path.name] = np.zeros(self.embedding_model.embedding_dim, dtype=np.float32)
        
        return embeddings
    
    def generate_dynamic_embeddings(self, clip_path: Path, frame_interval: int = 30) -> List[np.ndarray]:
        """
        Generate embeddings at multiple points in clip (for more granular matching).
        
        Args:
            clip_path: Path to video clip
            frame_interval: Extract frame every N frames
        
        Returns:
            List of embeddings at different timestamps
        """
        
        cap = cv2.VideoCapture(str(clip_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        frame_idx = 0
        
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frames.append(frame)
            frame_idx += frame_interval
        
        cap.release()
        
        if not frames:
            return [np.zeros(self.embedding_model.embedding_dim, dtype=np.float32)]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode_frames_batch(frames)
        return [e.astype(np.float32) for e in embeddings]


class ContextEmbedding:
    """Generate embeddings for song context (mood, genre, etc.)"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def embed_song_context(self, context) -> np.ndarray:
        """
        Create embedding from song context metadata.
        
        Args:
            context: SongContext object
        
        Returns:
            Embedding vector
        """
        
        # Create descriptive text
        text = f"{context.mood} {context.genre} music"
        text += f" at {context.bpm:.0f} BPM"
        text += f" with {context.danceability:.1%} danceability"
        
        # Generate embedding
        embedding = self.embedding_model.encode_text([text])[0]
        return embedding.astype(np.float32)
    
    def embed_clip_description(self, description: str) -> np.ndarray:
        """
        Generate embedding for text description of a clip.
        
        Args:
            description: Text description (e.g., "fast paced action scene")
        
        Returns:
            Embedding vector
        """
        
        embedding = self.embedding_model.encode_text([description])[0]
        return embedding.astype(np.float32)
    
    def embed_multiple_descriptions(self, descriptions: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple descriptions.
        
        Args:
            descriptions: List of text descriptions
        
        Returns:
            Array of embeddings
        """
        
        embeddings = self.embedding_model.encode_text(descriptions)
        return embeddings.astype(np.float32)
