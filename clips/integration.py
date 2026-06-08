"""Clip Information Integration with Main Pipeline"""

from pathlib import Path
from typing import Optional, List, Dict
from clips.information_system import ClipInformationSystem, ClipAnalyzer, ClipDatabase
from clips.pool_advanced import IntelligentClipPool
from config.settings import get_config


def integrate_clip_information_system(pipeline_instance) -> None:
    """
    Integrate advanced clip information system into main pipeline.
    
    This function enhances the pipeline with:
    - Comprehensive clip metadata extraction
    - Intelligent clip selection based on song context
    - Pool statistics and reporting
    - Usage tracking and recommendations
    """
    
    config = get_config()
    
    # Initialize information system
    print("[Pipeline] Initializing Clip Information System...")
    
    # Create intelligent pool
    pool = IntelligentClipPool(
        pool_dir=config.clips_dir,
        db_path=Path(config.output_dir) / "clips.db",
        embedding_model=pipeline_instance.embedding_model,
        use_cache=True
    )
    
    # Attach to pipeline
    pipeline_instance.clip_pool = pool
    pipeline_instance.clip_info_system = pool.info_system
    
    # Print pool report
    print("\n[Pipeline] Clip Pool Summary:")
    pool.print_pool_report()
    
    return pool


def select_clip_with_intelligence(pool: IntelligentClipPool,
                                 song_context: Dict,
                                 segment_duration: float,
                                 motion_preference: Optional[str] = None,
                                 color_preference: Optional[str] = None,
                                 avoid_clips: Optional[List[str]] = None) -> Optional[tuple]:
    """
    Intelligently select a clip using the advanced system.
    
    Args:
        pool: IntelligentClipPool instance
        song_context: Song analysis context
        segment_duration: Desired clip duration
        motion_preference: "static", "dynamic", or None
        color_preference: Color temperature preference
        avoid_clips: Clips to avoid
    
    Returns:
        (clip_path, metadata) or None
    """
    
    return pool.select_clip_for_segment(
        song_context=song_context,
        segment_duration=segment_duration,
        avoid_clips=avoid_clips,
        motion_preference=motion_preference,
        color_preference=color_preference
    )


def get_clip_recommendations(pool: IntelligentClipPool,
                            song_context: Dict,
                            num_recommendations: int = 5) -> List[Dict]:
    """
    Get recommended clips for a song segment.
    
    Returns recommendations with scores and explanations.
    """
    
    return pool.get_recommendations(song_context, num_recommendations)
