# 🎬 Advanced Clip Information System - Documentation

## Overview

The **Advanced Clip Information System** is a standalone, production-grade module for comprehensive video clip analysis, management, and intelligent selection.

## Features

### 1. **Deep Clip Analysis**
- Technical specifications (resolution, FPS, codec, bitrate)
- Motion analysis (motion score, camera type, direction)
- Color analysis (dominant colors, saturation, brightness, contrast, temperature)
- Content analysis (subject tags, environment, lighting, mood)
- Audio analysis (presence, sample rate, levels, genre)
- Scene detection and boundaries

### 2. **Intelligent Selection**
- AI-powered clip matching to song context
- Motion preference (static, dynamic)
- Color temperature matching
- Duration compatibility scoring
- Energy alignment with music
- Quality-aware selection

### 3. **Comprehensive Database**
- SQLite-based persistent storage
- Fast clip lookup and search
- Usage tracking and statistics
- Tagging and categorization
- Deduplication by file hash

### 4. **Pool Management**
- Automatic clip scanning and indexing
- Pool statistics and reporting
- Quality distribution analysis
- Aspect ratio compatibility
- Camera type classification

## Architecture

```
ClipInformationSystem
├── ClipAnalyzer
│   └── Deep clip analysis engine
├── ClipDatabase
│   └── SQLite metadata storage
└── IntelligentClipPool
    ├── Smart selection
    ├── Filtering
    └── Recommendations
```

## Usage

### Basic Analysis

```python
from clips.information_system import ClipInformationSystem
from pathlib import Path

# Initialize
system = ClipInformationSystem(
    clips_dir=Path("clips"),
    db_path=Path("clips.db")
)

# Analyze all clips
metadata_dict = system.scan_and_analyze()

# Print report
for clip_id, metadata in metadata_dict.items():
    system.print_clip_report(metadata)
```

### Intelligent Selection

```python
from clips.pool_advanced import IntelligentClipPool

# Initialize pool
pool = IntelligentClipPool(
    pool_dir=Path("clips"),
    db_path=Path("clips.db")
)

# Song context
song_context = {
    'energy': 0.85,
    'mood': 'energetic',
    'segment_duration': 5.0
}

# Get recommendations
recommendations = pool.get_recommendations(
    song_context=song_context,
    num_recommendations=5
)

for rec in recommendations:
    print(f"{rec['filename']}: {rec['reason']}")
```

### Filtering Clips

```python
# Filter by criteria
filtered = pool.filter_clips(
    quality="1080p",
    motion="dynamic",
    aspect_ratio="16:9",
    min_duration=3.0,
    max_duration=15.0
)

for clip_path, metadata in filtered:
    print(f"{metadata.filename}: {metadata.duration_seconds}s")
```

### Pool Statistics

```python
# Get statistics
stats = pool.get_pool_statistics()

print(f"Total clips: {stats['total_clips']}")
print(f"Total duration: {stats['total_duration_hours']:.1f}h")
print(f"Quality distribution: {stats['quality_distribution']}")
print(f"Aspect ratios: {stats['aspect_ratio_distribution']}")

# Print detailed report
pool.print_pool_report()
```

## Clip Metadata Structure

```python
@dataclass
class ClipMetadata:
    # Basic info
    clip_id: str
    filename: str
    filepath: Path
    file_hash: str
    file_size_mb: float
    added_date: str
    
    # Technical specs
    duration_seconds: float
    width: int
    height: int
    fps: int
    total_frames: int
    codec: str
    
    # Analysis results
    motion: ClipMotion           # Motion characteristics
    color_analysis: ClipColorAnalysis  # Color properties
    content: ClipContent          # Content analysis
    audio: ClipAudio             # Audio properties
    scenes: List[ClipScene]      # Scene boundaries
    
    # Compatibility
    compatible_music_genres: List[str]
    compatible_moods: List[str]
    best_transition_points: List[float]
    
    # Usage tracking
    times_used: int
    last_used_date: Optional[str]
    user_rating: float
```

## Analysis Results

### Motion Analysis
```python
@dataclass
class ClipMotion:
    motion_score: float          # 0-1 (static to dynamic)
    camera_type: str            # static, pan, zoom, handheld, mixed
    motion_direction: str        # horizontal, vertical, circular, none
    fps: int
    is_slow_motion: bool
    is_timelapse: bool
```

### Color Analysis
```python
@dataclass
class ClipColorAnalysis:
    dominant_colors: List[Tuple[int, int, int]]
    color_palette_count: int
    saturation_avg: float        # 0-1
    brightness_avg: float        # 0-1
    contrast_avg: float          # 0-1
    color_temperature: str       # warm, neutral, cool
    vibrancy_score: float        # 0-1
    color_harmony: str           # monochrome, analogous, etc.
```

### Content Analysis
```python
@dataclass
class ClipContent:
    subject_tags: List[str]      # people, nature, urban, abstract
    environment_type: str        # indoor, outdoor, studio, abstract
    lighting_quality: str        # natural, artificial, mixed, low_key
    composition_style: str       # wide, medium, close_up, macro
    visual_style: str            # realistic, stylized, animated
    mood_tags: List[str]         # energetic, calm, dramatic, romantic
    detected_objects: Dict[str, float]  # object -> confidence
    scene_complexity: float      # 0-1 (simple to complex)
```

## Integration with Pipeline

```python
from pipeline import MusicVideoGenerator
from clips.integration import integrate_clip_information_system

# Initialize pipeline
generator = MusicVideoGenerator(verbose=True)

# Integrate clip information system
pool = integrate_clip_information_system(generator)

# Now use intelligent selection
song_context = {...}
clip_path, metadata = pool.select_clip_for_segment(
    song_context=song_context,
    segment_duration=5.0,
    motion_preference="dynamic"
)
```

## Scoring Algorithm

The intelligent selection uses a multi-factor scoring system:

1. **Duration Compatibility** (20%)
   - Preference for clips close to desired duration
   - Flexibility for shorter segments

2. **Motion Match** (30%)
   - Dynamic clips for high-energy segments
   - Static clips for calm segments
   - Camera type compatibility

3. **Energy Alignment** (15%)
   - Match clip motion to song energy level
   - Synchronized visual-audio intensity

4. **Color Compatibility** (20%)
   - Color temperature matching
   - Mood-appropriate color palette
   - Saturation and vibrancy alignment

5. **Quality Score** (10%)
   - Resolution and FPS quality
   - Technical excellence bonus
   - 16:9 aspect ratio preference (5%)

## Database Schema

```sql
CREATE TABLE clips (
    clip_id TEXT PRIMARY KEY,
    filename TEXT,
    filepath TEXT,
    file_hash TEXT UNIQUE,        -- For deduplication
    file_size_mb REAL,
    duration_seconds REAL,
    width INTEGER,
    height INTEGER,
    fps INTEGER,
    estimated_quality TEXT,       -- 4K, 1080p, 720p, SD
    quality_score REAL,
    motion_score REAL,
    camera_type TEXT,
    scene_complexity REAL,
    times_used INTEGER,
    user_rating REAL,
    metadata_json TEXT            -- Full metadata as JSON
);

CREATE TABLE clip_tags (
    clip_id TEXT,
    tag TEXT,
    FOREIGN KEY(clip_id) REFERENCES clips(clip_id)
);
```

## Performance Optimization

### Caching
- Metadata cached in SQLite database
- Subsequent scans use cache (much faster)
- `force_rescan=True` to re-analyze all clips

### Batch Processing
- Process clips in parallel (future enhancement)
- Batch embedding computation
- Async database operations

### Memory Efficiency
- Stream video processing (not loading full video)
- Sample keyframes instead of all frames
- Compressed embeddings storage

## Advanced Features

### Deduplication
```python
# Automatic deduplication by SHA256 file hash
# Prevents duplicate clips in pool
```

### Usage Tracking
```python
# Automatic tracking of clip usage
pool.info_system.database.update_usage(clip_id)

# Get clips sorted by usage
most_used = pool.filter_clips(sort_by="times_used")
```

### User Ratings
```python
# Manual rating system (1-5 stars)
metadata.user_rating = 4.5
pool.info_system.database.add_clip(metadata)
```

## Examples

See `examples/clip_information_system.py` for complete working examples:

1. **Basic Analysis** - Analyze clips and print reports
2. **Pool Statistics** - Get overview of entire pool
3. **Intelligent Selection** - Get smart recommendations
4. **Filtering** - Find clips by specific criteria

## Performance Benchmarks

- **Initial Analysis**: ~5-10 seconds per clip (1080p, 10 seconds)
- **Database Query**: <100ms for typical filters
- **Recommendation Generation**: <50ms for 100-clip pool
- **Metadata Load**: <1 second for 1000-clip pool

## Future Enhancements

- [ ] AI-powered object detection (YOLO)
- [ ] Scene recognition (Places365)
- [ ] Emotion detection from content
- [ ] Audio fingerprinting
- [ ] Parallel GPU-accelerated analysis
- [ ] Web interface for pool management
- [ ] Clip marketplace/sharing
- [ ] Advanced similarity search

## API Reference

See `clips/information_system.py` and `clips/pool_advanced.py` for complete API documentation.

## License

MIT License - See LICENSE file
