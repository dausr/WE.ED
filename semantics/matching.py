"""Semantic Matching - Match clips to song segments based on embeddings"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import euclidean, cosine
from dataclasses import dataclass


@dataclass
class MatchResult:
    """Result of semantic matching"""
    segment_idx: int
    clip_idx: int
    similarity_score: float
    color_match: float
    motion_match: float
    combined_score: float
    match_type: str


class SemanticMatcher:
    """Match clips to song segments using multiple similarity metrics"""
    
    def __init__(self, embedding_weight: float = 0.7, 
                 feature_weight: float = 0.3):
        self.embedding_weight = embedding_weight
        self.feature_weight = feature_weight
    
    def match_clips_to_segments(self,
                               segment_embeddings: List[np.ndarray],
                               clip_embeddings: List[np.ndarray],
                               segment_features: Optional[List[Dict]] = None,
                               clip_features: Optional[List[Dict]] = None,
                               num_matches: int = 1) -> List[List[MatchResult]]:
        """
        Match clips to song segments.
        
        Args:
            segment_embeddings: Embeddings for song segments
            clip_embeddings: Embeddings for video clips
            segment_features: Optional features for each segment
            clip_features: Optional features for each clip
            num_matches: Number of clips to match per segment
        
        Returns:
            List of match results for each segment
        """
        
        all_matches = []
        
        for seg_idx, segment_emb in enumerate(segment_embeddings):
            segment_feat = segment_features[seg_idx] if segment_features else None
            
            # Score all clips against this segment
            clip_scores = []
            
            for clip_idx, clip_emb in enumerate(clip_embeddings):
                clip_feat = clip_features[clip_idx] if clip_features else None
                
                # Embedding similarity
                emb_sim = self._embedding_similarity(segment_emb, clip_emb)
                
                # Feature similarity
                feat_sim = 0.0
                if segment_feat and clip_feat:
                    feat_sim = self._feature_similarity(segment_feat, clip_feat)
                
                # Combined score
                combined = (
                    self.embedding_weight * emb_sim +
                    self.feature_weight * feat_sim
                )
                
                clip_scores.append((clip_idx, emb_sim, feat_sim, combined))
            
            # Sort by combined score
            clip_scores.sort(key=lambda x: x[3], reverse=True)
            
            # Create results
            segment_matches = []
            for rank, (clip_idx, emb_sim, feat_sim, combined) in enumerate(clip_scores[:num_matches]):
                match_type = self._determine_match_type(emb_sim, feat_sim)
                
                result = MatchResult(
                    segment_idx=seg_idx,
                    clip_idx=clip_idx,
                    similarity_score=float(emb_sim),
                    color_match=float(feat_sim) if clip_features and segment_features else 0.0,
                    motion_match=0.0,  # Can be computed separately
                    combined_score=float(combined),
                    match_type=match_type
                )
                segment_matches.append(result)
            
            all_matches.append(segment_matches)
        
        return all_matches
    
    def _embedding_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between embeddings.
        Returns value in [0, 1] where 1 is identical.
        """
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Normalize
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(max(0.0, similarity))  # Clamp to [0, 1]
    
    def _feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Compute similarity based on visual features.
        """
        
        scores = []
        
        # Color histogram similarity (if available)
        if 'color_histogram' in features1 and 'color_histogram' in features2:
            hist_sim = self._histogram_similarity(
                features1['color_histogram'],
                features2['color_histogram']
            )
            scores.append(hist_sim)
        
        # Brightness/saturation match
        if 'brightness' in features1 and 'brightness' in features2:
            brightness_sim = 1.0 - abs(features1['brightness'] - features2['brightness'])
            scores.append(brightness_sim)
        
        if 'saturation' in features1 and 'saturation' in features2:
            saturation_sim = 1.0 - abs(features1['saturation'] - features2['saturation'])
            scores.append(saturation_sim)
        
        # Motion similarity
        if 'motion_score' in features1 and 'motion_score' in features2:
            motion_sim = 1.0 - abs(features1['motion_score'] - features2['motion_score'])
            scores.append(motion_sim)
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _histogram_similarity(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """
        Compare color histograms using chi-square distance.
        """
        
        if hist1 is None or hist2 is None:
            return 0.0
        
        # Normalize histograms
        hist1 = hist1 / (np.sum(hist1) + 1e-8)
        hist2 = hist2 / (np.sum(hist2) + 1e-8)
        
        # Chi-square distance
        chi_square = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-8))
        similarity = np.exp(-chi_square)
        
        return float(similarity)
    
    def _determine_match_type(self, embedding_sim: float, feature_sim: float) -> str:
        """
        Determine type of match based on scores.
        """
        
        if embedding_sim > 0.8:
            return "semantic_match"
        elif feature_sim > 0.7:
            return "feature_match"
        elif embedding_sim > 0.6:
            return "acceptable_match"
        else:
            return "fallback_match"
    
    def filter_matches(self, matches: List[List[MatchResult]], 
                      min_score: float = 0.5) -> List[List[MatchResult]]:
        """
        Filter matches by minimum score threshold.
        """
        
        filtered = []
        for segment_matches in matches:
            segment_filtered = [
                m for m in segment_matches 
                if m.combined_score >= min_score
            ]
            filtered.append(segment_filtered)
        
        return filtered
    
    def rerank_with_diversity(self, matches: List[List[MatchResult]],
                             clip_embeddings: List[np.ndarray],
                             diversity_penalty: float = 0.2) -> List[List[MatchResult]]:
        """
        Rerank matches to encourage diversity (avoid using same clip too often).
        """
        
        reranked = []
        clip_usage_count = {}
        
        for segment_matches in matches:
            segment_reranked = []
            
            for match in segment_matches:
                # Penalty based on previous usage
                usage_count = clip_usage_count.get(match.clip_idx, 0)
                diversity_score = match.combined_score * (1 - diversity_penalty * usage_count)
                
                # Create new match with updated score
                new_match = MatchResult(
                    segment_idx=match.segment_idx,
                    clip_idx=match.clip_idx,
                    similarity_score=match.similarity_score,
                    color_match=match.color_match,
                    motion_match=match.motion_match,
                    combined_score=float(diversity_score),
                    match_type=match.match_type
                )
                segment_reranked.append(new_match)
            
            # Sort by new score
            segment_reranked.sort(key=lambda m: m.combined_score, reverse=True)
            
            # Update usage count for selected clip
            if segment_reranked:
                clip_usage_count[segment_reranked[0].clip_idx] = \
                    clip_usage_count.get(segment_reranked[0].clip_idx, 0) + 1
            
            reranked.append(segment_reranked)
        
        return reranked
