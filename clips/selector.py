"""Semantic Clip Selector - AI-Powered Intelligent Clip Matching"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.spatial.distance import cosine
import random


@dataclass
class SelectionResult:
    """Result of clip selection"""
    clip_idx: int
    similarity_score: float
    diversity_score: float
    combined_score: float
    reason: str


class SemanticClipSelector:
    """Select clips based on semantic similarity to song context"""
    
    def __init__(self, embedding_dim: int = 512, diversity_weight: float = 0.3):
        self.embedding_dim = embedding_dim
        self.diversity_weight = diversity_weight  # Balance between similarity and diversity
    
    def select_clips_for_timeline(self, 
                                  song_embedding: np.ndarray,
                                  song_segments: List[Dict],
                                  clip_pool: List[np.ndarray],
                                  clips_per_segment: int = 1,
                                  avoid_repeats: bool = True,
                                  max_repeat_distance: float = 5.0) -> List[Tuple[int, float]]:
        """
        Select clips for each segment of the song.
        
        Args:
            song_embedding: Overall song semantic embedding
            song_segments: List of segments with their embeddings [{embedding: np.ndarray, timestamp: float}]
            clip_pool: List of clip embeddings
            clips_per_segment: Number of clips to select per segment
            avoid_repeats: Whether to avoid selecting same clip too frequently
            max_repeat_distance: Minimum time (seconds) between same clip repetitions
        
        Returns:
            List of (clip_idx, score) tuples
        """
        
        selections = []
        recently_used = {}  # {clip_idx: last_timestamp}
        
        for segment_idx, segment in enumerate(song_segments):
            segment_embedding = segment['embedding']
            segment_timestamp = segment.get('timestamp', 0)
            
            # Find best clips for this segment
            candidates = self._find_best_candidates(
                segment_embedding,
                clip_pool,
                k=min(clips_per_segment * 3, len(clip_pool))  # Get top candidates
            )
            
            # Filter based on repeat avoidance
            if avoid_repeats:
                filtered_candidates = [
                    (idx, score) for idx, score in candidates
                    if self._can_use_clip(idx, segment_timestamp, recently_used, max_repeat_distance)
                ]
                if not filtered_candidates:
                    filtered_candidates = candidates  # Fallback if all filtered
            else:
                filtered_candidates = candidates
            
            # Select clips
            for i in range(min(clips_per_segment, len(filtered_candidates))):
                clip_idx, score = filtered_candidates[i]
                selections.append((clip_idx, float(score)))
                recently_used[clip_idx] = segment_timestamp
        
        return selections
    
    def _find_best_candidates(self, 
                             query_embedding: np.ndarray,
                             clip_embeddings: List[np.ndarray],
                             k: int = 10) -> List[Tuple[int, float]]:
        """
        Find top-k most similar clips to query embedding.
        Returns list of (clip_idx, similarity_score) sorted by similarity.
        """
        
        if not clip_embeddings:
            return []
        
        similarities = []
        for idx, clip_emb in enumerate(clip_embeddings):
            if clip_emb is None:
                similarity = 0.0
            else:
                # Cosine similarity (higher is better)
                similarity = 1 - cosine(query_embedding, clip_emb)
            similarities.append((idx, float(similarity)))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def _can_use_clip(self, clip_idx: int, current_time: float, 
                     recently_used: Dict[int, float], max_repeat_distance: float) -> bool:
        """
        Check if clip can be used (hasn't been used too recently)
        """
        
        if clip_idx not in recently_used:
            return True
        
        last_used = recently_used[clip_idx]
        time_since_use = current_time - last_used
        
        return time_since_use >= max_repeat_distance
    
    def rerank_with_diversity(self, 
                             selections: List[Tuple[int, float]],
                             clip_embeddings: List[np.ndarray],
                             diversity_factor: float = 0.3) -> List[Tuple[int, float, float]]:
        """
        Rerank selections to balance similarity and diversity.
        Penalizes clips that are too similar to recently selected ones.
        
        Returns list of (clip_idx, original_score, diversity_score)
        """
        
        reranked = []
        selected_embeddings = []
        
        for clip_idx, similarity_score in selections:
            if clip_embeddings[clip_idx] is None:
                diversity_score = 1.0
            else:
                # Compute diversity: average distance to previously selected clips
                if selected_embeddings:
                    distances = [
                        cosine(clip_embeddings[clip_idx], selected_emb)
                        for selected_emb in selected_embeddings
                    ]
                    diversity_score = float(np.mean(distances))
                else:
                    diversity_score = 1.0  # First clip has max diversity
            
            # Combined score: balance between similarity and diversity
            combined_score = (
                (1 - diversity_factor) * similarity_score +
                diversity_factor * diversity_score
            )
            
            reranked.append((clip_idx, float(similarity_score), float(diversity_score)))
            selected_embeddings.append(clip_embeddings[clip_idx])
        
        return reranked
    
    def select_with_context(self,
                           segment_embedding: np.ndarray,
                           clip_embeddings: List[np.ndarray],
                           clip_metadata: Optional[List[Dict]] = None,
                           context_tags: Optional[List[str]] = None,
                           num_selections: int = 1) -> List[SelectionResult]:
        """
        Advanced selection considering clip metadata and context tags.
        
        Args:
            segment_embedding: Semantic embedding of song segment
            clip_embeddings: List of clip embeddings
            clip_metadata: Optional metadata for each clip
            context_tags: Context tags (e.g., ['energetic', 'dark', 'bright'])
            num_selections: Number of clips to select
        
        Returns:
            List of SelectionResult objects
        """
        
        candidates = self._find_best_candidates(segment_embedding, clip_embeddings, k=len(clip_embeddings))
        
        # Apply metadata filters if provided
        if clip_metadata and context_tags:
            candidates = self._filter_by_metadata(candidates, clip_metadata, context_tags)
        
        results = []
        for i, (clip_idx, similarity_score) in enumerate(candidates[:num_selections]):
            reason = f"Top match (sim: {similarity_score:.3f})"
            
            if clip_metadata and clip_idx < len(clip_metadata):
                reason = f"Match with tags: {context_tags}"
            
            result = SelectionResult(
                clip_idx=clip_idx,
                similarity_score=float(similarity_score),
                diversity_score=1.0 if i == 0 else float(1.0 - (i / len(candidates))),
                combined_score=float(similarity_score),
                reason=reason
            )
            results.append(result)
        
        return results
    
    def _filter_by_metadata(self, candidates: List[Tuple[int, float]], 
                           clip_metadata: List[Dict], 
                           context_tags: List[str]) -> List[Tuple[int, float]]:
        """
        Filter candidates based on metadata tags matching context.
        """
        
        if not context_tags:
            return candidates
        
        filtered = []
        for clip_idx, score in candidates:
            if clip_idx < len(clip_metadata):
                clip_tags = clip_metadata[clip_idx].get('tags', [])
                tag_matches = sum(1 for tag in context_tags if tag in clip_tags)
                if tag_matches > 0:
                    # Boost score based on tag matches
                    boosted_score = score * (1 + 0.1 * tag_matches)
                    filtered.append((clip_idx, boosted_score))
        
        return sorted(filtered, key=lambda x: x[1], reverse=True) if filtered else candidates
    
    def suggest_clip_pool_diversity(self, clip_embeddings: List[np.ndarray], 
                                   max_cluster_size: int = 5) -> Dict[int, str]:
        """
        Suggest which clips from pool are most diverse (good for variety).
        
        Returns mapping of clip_idx to cluster_id
        """
        
        if not clip_embeddings or all(e is None for e in clip_embeddings):
            return {}
        
        # Simple clustering: divide embeddings into groups
        valid_embeddings = [(i, e) for i, e in enumerate(clip_embeddings) if e is not None]
        
        if not valid_embeddings:
            return {}
        
        clusters = {}
        cluster_id = 0
        
        # Sort by diversity
        for idx, embedding in valid_embeddings:
            if not clusters:
                clusters[idx] = cluster_id
            else:
                # Check distance to existing clusters
                distances = [
                    min(cosine(embedding, valid_embeddings[c_idx][1]) 
                        for c_idx in range(len(valid_embeddings)) 
                        if clusters.get(valid_embeddings[c_idx][0]) == cid)
                    for cid in set(clusters.values())
                ]
                
                # Assign to most distant cluster or create new one
                if max(distances) > 0.3:  # Diversity threshold
                    assigned_cluster = distances.index(max(distances))
                else:
                    assigned_cluster = max(set(clusters.values())) + 1 if clusters else 0
                    if assigned_cluster >= max_cluster_size:
                        assigned_cluster = 0
                
                clusters[idx] = assigned_cluster
        
        return clusters
