"""Example - Using Advanced Clip Information System"""

from pathlib import Path
from clips.information_system import ClipInformationSystem
from clips.pool_advanced import IntelligentClipPool


def example_basic_analysis():
    """
    Example 1: Basic clip analysis and reporting
    """
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Clip Analysis")
    print("="*70)
    
    # Initialize system
    system = ClipInformationSystem(
        clips_dir=Path("clips"),
        db_path=Path("example_clips.db")
    )
    
    # Scan and analyze clips
    metadata_dict = system.scan_and_analyze()
    
    # Print report for first clip
    if metadata_dict:
        first_clip = list(metadata_dict.values())[0]
        system.print_clip_report(first_clip)


def example_pool_statistics():
    """
    Example 2: Get pool statistics and overview
    """
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Pool Statistics")
    print("="*70)
    
    # Initialize pool
    pool = IntelligentClipPool(
        pool_dir=Path("clips"),
        db_path=Path("example_clips.db")
    )
    
    # Print pool report
    pool.print_pool_report()


def example_intelligent_selection():
    """
    Example 3: Intelligent clip selection based on song context
    """
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Intelligent Clip Selection")
    print("="*70)
    
    # Initialize pool
    pool = IntelligentClipPool(
        pool_dir=Path("clips"),
        db_path=Path("example_clips.db")
    )
    
    # Define song context
    song_context = {
        'genre': 'electronic',
        'mood': 'energetic',
        'energy': 0.85,
        'segment_duration': 5.0
    }
    
    # Get intelligent recommendations
    recommendations = pool.get_recommendations(song_context, num_recommendations=3)
    
    print(f"\n📋 Top Recommendations for '{song_context['mood']} {song_context['genre']}' music:")
    print()
    
    for rec in recommendations:
        print(f"  #{rec['rank']} - {rec['filename']}")
        print(f"      Score: {rec['score']:.1%}")
        print(f"      Reason: {rec['reason']}")
        print()


def example_filtering():
    """
    Example 4: Filter clips by criteria
    """
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Filter Clips by Criteria")
    print("="*70)
    
    # Initialize pool
    pool = IntelligentClipPool(
        pool_dir=Path("clips"),
        db_path=Path("example_clips.db")
    )
    
    # Filter criteria
    print("\n🎬 Finding high-quality, dynamic clips in 16:9 format...")
    
    filtered = pool.filter_clips(
        quality="1080p",
        motion="dynamic",
        aspect_ratio="16:9",
        min_duration=3.0,
        max_duration=15.0
    )
    
    print(f"\nFound {len(filtered)} matching clips:\n")
    
    for clip_path, metadata in filtered[:5]:  # Show first 5
        print(f"  📹 {metadata.filename}")
        print(f"      Duration: {metadata.duration_seconds:.1f}s")
        print(f"      Quality: {metadata.estimated_quality}")
        print(f"      Motion: {metadata.motion.motion_score:.0%}")
        print(f"      Camera: {metadata.motion.camera_type}")
        print()


if __name__ == "__main__":
    # Run examples
    example_basic_analysis()
    example_pool_statistics()
    example_intelligent_selection()
    example_filtering()
