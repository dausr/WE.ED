"""Advanced Clip Pool - Enhanced with Intelligence System"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from scipy.spatial.distance import cosine
import numpy as np
from dataclasses import asdict

from clips.information_system import (
    ClipInformationSystem,
    ClipAnalyzer,
    ClipDatabase,
    ClipMetadata
)


class IntelligentClipPool:
    """Advanced clip pool with intelligent matching and management"""
    
    def __init__(self, pool_dir: Path, db_path: Path = Path("clips.db"), 
                 embedding_model=None, use_cache: bool = True):
        self.pool_dir = Path(pool_dir)
        self.embedding_model = embedding_model
        
        # Initialize information system
        self.info_system = ClipInformationSystem(
            clips_dir=pool_dir,
            db_path=db_path,
            embedding_model=embedding_model
        )
        
        # Load or analyze clips
        self.clips_metadata: Dict[str, ClipMetadata] = {}
        self._load_clips(use_cache)
    
    def _load_clips(self, use_cache: bool = True) -> None:
        """Load clip information from database or analyze"""
        
        if use_cache:
            # Try loading from existing database
            try:
                # Simple check - if database has clips, load them
                self.clips_metadata = self.info_system.database.search_clips({})
                if self.clips_metadata:
                    print(f"[Clip Pool] Loaded {len(self.clips_metadata)} clips from cache")
                    return
            except:
                pass
        
        # Analyze all clips
        print(f"[Clip Pool] Scanning and analyzing clips in {self.pool_dir}...")
        self.clips_metadata = self.info_system.scan_and_analyze()
    
    def select_clip_for_segment(self, 
                               song_context: Dict,
                               segment_duration: float,
                               avoid_clips: Optional[List[str]] = None,
                               motion_preference: Optional[str] = None,
                               color_preference: Optional[str] = None) -> Optional[Tuple[str, ClipMetadata]]:
        """
        Intelligently select a clip for a song segment.
        
        Args:
            song_context: Song analysis (mood, energy, genre, etc.)
            segment_duration: Desired clip duration
            avoid_clips: List of clip IDs to avoid
            motion_preference: "static", "dynamic", or None
            color_preference: Color temperature preference
        
        Returns:
            (clip_path, metadata) or None if no match
        """
        
        if not self.clips_metadata:
            return None
        
        avoid_clips = avoid_clips or []
        
        # Score all clips
        scored_clips = []
        
        for clip_id, metadata in self.clips_metadata.items():
            if clip_id in avoid_clips:
                continue
            
            # Score based on multiple factors
            score = self._score_clip(
                metadata,
                song_context,
                segment_duration,
                motion_preference,
                color_preference
            )
            
            scored_clips.append((score, clip_id, metadata))
        
        if not scored_clips:
            return None
        
        # Return best match
        scored_clips.sort(key=lambda x: x[0], reverse=True)
        best_score, best_id, best_metadata = scored_clips[0]
        
        return (str(best_metadata.filepath), best_metadata)
    
    def _score_clip(self,
                   metadata: ClipMetadata,
                   song_context: Dict,
                   segment_duration: float,
                   motion_preference: Optional[str] = None,
                   color_preference: Optional[str] = None) -> float:
        """
        Score a clip for relevance to song segment.
        
        Scoring factors:
        - Duration compatibility (0-0.2)
        - Motion match (0-0.3)
        - Color compatibility (0-0.2)
        - Energy alignment (0-0.2)
        - Quality score (0-0.1)
        """
        
        score = 0.0
        
        # Duration compatibility (prefer clips close to segment duration)
        duration_diff = abs(metadata.duration_seconds - segment_duration)
        if duration_diff < segment_duration * 0.5:
            duration_score = 1.0 - (duration_diff / segment_duration)
        else:
            duration_score = max(0, 1.0 - duration_diff / 60)
        score += duration_score * 0.2
        
        # Motion preference
        if motion_preference == "dynamic" and metadata.motion.motion_score > 0.5:
            score += 0.15
        elif motion_preference == "static" and metadata.motion.motion_score < 0.5:
            score += 0.15
        else:
            score += 0.075
        
        # Energy alignment (if provided)
        if 'energy' in song_context:
            energy_diff = abs(metadata.motion.motion_score - song_context['energy'])
            energy_score = 1.0 - energy_diff
            score += energy_score * 0.15
        
        # Color compatibility
        if color_preference and metadata.color_analysis.color_temperature == color_preference:
            score += 0.2
        else:
            score += 0.1
        
        # Quality bonus
        score += metadata.quality_score * 0.1
        
        # Aspect ratio preference (16:9 preferred)
        if metadata.aspect_ratio_label == "16:9":
            score += 0.05
        
        return score
    
    def filter_clips(self, **criteria) -> List[Tuple[str, ClipMetadata]]:
        """
        Filter clips by multiple criteria.
        
        Criteria:
        - quality: "4K", "1080p", "720p", "SD"
        - motion: "static", "dynamic", "moderate"
        - color_temp: "warm", "cool", "neutral"
        - min_duration, max_duration
        - aspect_ratio: "16:9", "4:3", etc.
        """
        
        results = []
        
        for clip_id, metadata in self.clips_metadata.items():
            if self._matches_criteria(metadata, criteria):
                results.append((str(metadata.filepath), metadata))
        
        return results
    
    def _matches_criteria(self, metadata: ClipMetadata, criteria: Dict) -> bool:
        """Check if clip matches filter criteria"""
        
        if 'quality' in criteria and metadata.estimated_quality != criteria['quality']:
            return False
        
        if 'aspect_ratio' in criteria and metadata.aspect_ratio_label != criteria['aspect_ratio']:
            return False
        
        if 'min_duration' in criteria and metadata.duration_seconds < criteria['min_duration']:
            return False
        
        if 'max_duration' in criteria and metadata.duration_seconds > criteria['max_duration']:
            return False
        
        if 'motion' in criteria:
            motion_pref = criteria['motion']
            if motion_pref == "static" and metadata.motion.motion_score > 0.5:
                return False
            elif motion_pref == "dynamic" and metadata.motion.motion_score < 0.5:
                return False
        
        if 'color_temp' in criteria and metadata.color_analysis.color_temperature != criteria['color_temp']:
            return False
        
        if 'camera_type' in criteria and metadata.motion.camera_type != criteria['camera_type']:
            return False
        
        if 'min_quality_score' in criteria and metadata.quality_score < criteria['min_quality_score']:
            return False
        
        return True
    
    def get_recommendations(self, song_context: Dict, num_recommendations: int = 5) -> List[Dict]:
        """
        Get recommended clips based on song analysis.
        
        Returns list of recommendations with explanations.
        """
        
        recommendations = []
        
        # Determine clip preferences from song context
        if song_context.get('energy', 0) > 0.7:
            motion_pref = "dynamic"
        else:
            motion_pref = "static"
        
        # Filter by preferences
        filtered = self.filter_clips(motion=motion_pref)
        
        # Score and sort
        scored = []
        for clip_path, metadata in filtered:
            score = self._score_clip(
                metadata,
                song_context,
                song_context.get('segment_duration', 5.0),
                motion_preference=motion_pref
            )
            scored.append((score, clip_path, metadata))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Create recommendations
        for i, (score, clip_path, metadata) in enumerate(scored[:num_recommendations]):
            recommendation = {
                'rank': i + 1,
                'clip_path': clip_path,
                'clip_id': metadata.clip_id,
                'filename': metadata.filename,
                'score': float(score),
                'reason': self._get_recommendation_reason(metadata, song_context),
                'metadata': asdict(metadata) if hasattr(metadata, '__dict__') else metadata
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _get_recommendation_reason(self, metadata: ClipMetadata, song_context: Dict) -> str:
        """Generate human-readable reason for recommendation"""
        
        reasons = []
        
        if song_context.get('energy', 0) > 0.7 and metadata.motion.motion_score > 0.5:
            reasons.append(f"High energy match ({metadata.motion.motion_score:.0%} motion)")
        
        if metadata.aspect_ratio_label == "16:9":
            reasons.append("Perfect 16:9 aspect ratio")
        
        if metadata.estimated_quality in ["4K", "1080p"]:
            reasons.append(f"High quality ({metadata.estimated_quality})")
        
        if metadata.quality_score > 0.8:
            reasons.append("Excellent technical quality")
        
        if not reasons:
            reasons.append("Good overall match")
        
        return " • ".join(reasons)
    
    def get_pool_statistics(self) -> Dict:
        """Get statistics about the clip pool"""
        
        if not self.clips_metadata:
            return {}
        
        metadata_list = list(self.clips_metadata.values())
        
        durations = [m.duration_seconds for m in metadata_list]
        qualities = [m.estimated_quality for m in metadata_list]
        motion_scores = [m.motion.motion_score for m in metadata_list]
        
        stats = {
            'total_clips': len(self.clips_metadata),
            'total_duration_hours': sum(durations) / 3600,
            'avg_duration': np.mean(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'quality_distribution': self._count_qualities(qualities),
            'avg_motion_score': float(np.mean(motion_scores)),
            'aspect_ratio_distribution': self._count_aspect_ratios(metadata_list),
            'camera_types': self._count_camera_types(metadata_list),
            '16_9_clips': len([m for m in metadata_list if m.aspect_ratio_label == "16:9"]),
            'high_quality_clips': len([m for m in metadata_list if m.quality_score > 0.8]),
        }
        
        return stats
    
    def _count_qualities(self, qualities: List[str]) -> Dict[str, int]:
        """Count clips by quality tier"""
        counts = {}
        for q in qualities:
            counts[q] = counts.get(q, 0) + 1
        return counts
    
    def _count_aspect_ratios(self, metadata_list: List[ClipMetadata]) -> Dict[str, int]:
        """Count clips by aspect ratio"""
        counts = {}
        for m in metadata_list:
            label = m.aspect_ratio_label
            counts[label] = counts.get(label, 0) + 1
        return counts
    
    def _count_camera_types(self, metadata_list: List[ClipMetadata]) -> Dict[str, int]:
        """Count clips by camera type"""
        counts = {}
        for m in metadata_list:
            cam_type = m.motion.camera_type
            counts[cam_type] = counts.get(cam_type, 0) + 1
        return counts
    
    def print_pool_report(self) -> None:
        """Print detailed pool analysis report"""
        
        stats = self.get_pool_statistics()
        
        if not stats:
            print("No clips in pool")
            return
        
        print(f"\n{'='*70}")
        print(f"CLIP POOL ANALYSIS REPORT")
        print(f"{'='*70}")
        
        print(f"\n📊 POOL STATISTICS:")
        print(f"  Total Clips: {stats['total_clips']}")
        print(f"  Total Duration: {stats['total_duration_hours']:.1f} hours")
        print(f"  Average Clip Duration: {stats['avg_duration']:.1f}s")
        print(f"  Duration Range: {stats['min_duration']:.1f}s - {stats['max_duration']:.1f}s")
        
        print(f"\n🎬 QUALITY DISTRIBUTION:")
        for quality, count in stats['quality_distribution'].items():
            print(f"  {quality}: {count} clips")
        
        print(f"\n📐 ASPECT RATIOS:")
        for aspect, count in stats['aspect_ratio_distribution'].items():
            print(f"  {aspect}: {count} clips")
        
        print(f"\n🎥 CAMERA TYPES:")
        for cam_type, count in stats['camera_types'].items():
            print(f"  {cam_type}: {count} clips")
        
        print(f"\n⚡ MOTION ANALYSIS:")
        print(f"  Average Motion Score: {stats['avg_motion_score']:.1%}")
        print(f"  High Quality Clips (>80%): {stats['high_quality_clips']}")
        print(f"  16:9 Compatible: {stats['16_9_clips']}")
        
        print(f"\n{'='*70}\n")
