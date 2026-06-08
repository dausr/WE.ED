# 📚 WE.ED Framework - Complete Usage Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Core Concepts](#core-concepts)
5. [Usage Examples](#usage-examples)
6. [Configuration](#configuration)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Usage

```bash
# Create directory structure
mkdir -p sounds clips output

# Place your MP3 in sounds/
cp your_song.mp3 sounds/

# Place your video clips in clips/
cp *.mp4 clips/

# Generate music video
python main.py --song sounds/your_song.mp3 --clips-dir clips --output output/
```

### Result

Your music video will be created at: `output/musicvideo_v001.mp4`

---

## Installation

### Requirements

- Python 3.8+
- FFmpeg and FFprobe installed
- CUDA-capable GPU (recommended for embeddings)

### Setup

```bash
# Clone or download the repository
cd WE.ED

# Install dependencies
pip install -r requirements.txt

# Optional: Download CLIP model (for better embeddings)
python -c "import clip; clip.load('clip-vit-base-patch32', device='cuda')"
```

---

## Project Structure

```
WE.ED/
├── audio/                  # Audio processing
│   ├── analyzer.py        # BPM, beat detection
│   └── metadata.py        # MP3 tag extraction
├── clips/                  # Clip management
│   ├── pool.py            # Clip indexing
│   ├── preprocessor.py    # Frame extraction
│   └── selector.py        # Semantic selection
├── semantics/              # AI analysis
│   ├── analyzer.py        # Song context
│   ├── embeddings.py      # Vector embeddings
│   └── matching.py        # Semantic matching
├── video/                  # Video generation
│   ├── editor.py          # FFmpeg integration
│   ├── transitions.py     # Effects
│   └── renderer.py        # Output rendering
├── config/                 # Configuration
│   ├── settings.py        # Config management
│   └── defaults.yaml      # Defaults
├── examples/               # Example configs
│   ├── config_energetic_dance.yaml
│   ├── config_calm_ambient.yaml
│   └── config_hiphop.yaml
├── tests/                  # Integration tests
├── pipeline.py            # Main orchestrator
└── main.py                # CLI entry point
```

---

## Core Concepts

### 1. **Audio Analysis**

The framework analyzes your MP3 to extract:
- **BPM** - Beats per minute
- **Beats** - Precise beat times
- **Genre** - Estimated from metadata and audio
- **Features** - Onset strength, spectral properties

### 2. **Song Context**

Extracted from metadata and audio:
- **Mood** - energetic, calm, melancholic, happy, aggressive, groovy
- **Energy Level** - 0-1 scale
- **Danceability** - 0-1 scale

### 3. **Clip Semantics**

Each clip is analyzed for:
- **Semantic embedding** - CLIP model representation
- **Color histogram** - Color distribution
- **Motion score** - Activity level
- **Brightness/Saturation** - Visual properties

### 4. **Intelligent Matching**

Clips are matched to song segments based on:
- **Semantic similarity** - AI understanding of visual-audio fit
- **Feature matching** - Color, motion, brightness alignment
- **Diversity** - Avoid repetitive clips
- **Aspect ratio** - Preserve 16:9 without black bars

### 5. **Transitions**

Professional effects between clips:
- **Cross-Fade** - Classic smooth blend
- **Wobble Zoom** - Bass-synchronized dynamic zoom
- **Wipe** - Progressive reveal
- **Stutter** - Beat-synced frame repetition
- **Pixelate** - Pixelization dissolve

---

## Usage Examples

### Example 1: Basic Dance Music Video

```bash
python main.py \
  --song "songs/dance_track.mp3" \
  --clips-dir "dance_clips" \
  --output "videos/"
```

### Example 2: With Custom Config

```bash
python main.py \
  --song "songs/ambient.mp3" \
  --clips-dir "footage" \
  --output "videos/" \
  --config examples/config_calm_ambient.yaml
```

### Example 3: Programmatic Usage

```python
from pipeline import MusicVideoGenerator
from config.settings import WEEDConfig

# Load config
config = WEEDConfig.from_yaml('examples/config_energetic_dance.yaml')

# Create generator
generator = MusicVideoGenerator(config=config, verbose=True)

# Generate video
output_path = generator.generate('songs/track.mp3')
print(f"Generated: {output_path}")
```

### Example 4: Batch Processing

```python
from pathlib import Path
from pipeline import MusicVideoGenerator

generator = MusicVideoGenerator(verbose=True)

# Process all MP3s in directory
for song_path in Path('songs').glob('*.mp3'):
    print(f"Processing {song_path.name}...")
    generator.generate(str(song_path))
```

---

## Configuration

### Audio Settings

```yaml
audio:
  sample_rate: 44100      # Audio sample rate (Hz)
  n_fft: 2048            # FFT window size
  hop_length: 512        # Hop length for STFT
  min_bpm: 80            # Minimum BPM estimate
  max_bpm: 180           # Maximum BPM estimate
```

### Video Settings

```yaml
video:
  resolution: "1920x1080"  # 16:9 standard
  frame_rate: 30           # Output FPS
  codec: "libx264"         # Video codec
  preset: "medium"         # Encoding speed (slower = better quality)
  bitrate: "8000k"         # Video bitrate
```

### Clip Selection

```yaml
clip_selection:
  avoid_repeats: true              # Don't use same clip twice nearby
  max_repeat_distance: 5           # Seconds between clip repeats
  semantic_threshold: 0.7          # Min similarity score (0-1)
  diversity_factor: 0.3            # Balance diversity vs similarity
```

### Transitions

```yaml
transitions:
  blend_fade: true          # Use blend effects
  fade_duration: 0.3        # Transition length (seconds)
  wobble_intensity: 0.8     # Wobble strength (0-1)
  wobble_frequency: 3.0     # Wobble speed (Hz-like)
  zoom_direction: "center_outward"  # Zoom direction
  stutter_effect: true      # Enable stutter
  stutter_frames: 3         # Frames per stutter
```

### Output Settings

```yaml
output:
  version_format: "{original_name}_v{version:03d}"
  auto_versioning: true      # Auto-increment versions
  preserve_aspect_ratio: true # Keep clip aspect
  no_black_bars: true        # Avoid black padding
  output_format: "mp4"       # Output format
```

### Semantics

```yaml
semantics:
  embedding_model: "clip-vit-base-patch32"  # CLIP model
  device: "cuda"             # GPU or CPU
  batch_size: 32             # Embedding batch size
```

---

## Advanced Usage

### Custom Mood Mapping

Modify `semantics/analyzer.py` to add custom mood keywords:

```python
self.mood_keywords = {
    'your_mood': ['keyword1', 'keyword2', 'keyword3']
}
```

### Custom Transition Effects

Add new transitions in `video/transitions.py`:

```python
def custom_transition(self, frame1, frame2, progress):
    # Your effect code here
    return result_frame
```

### Clip Filtering

Filter clips before processing:

```python
# Get only 16:9 clips
valid_clips = generator.clip_pool.get_16_9_clips()

# Filter by duration
short_clips = generator.clip_pool.get_by_duration(
    min_duration=2.0,
    max_duration=10.0
)
```

### Manual Timeline Creation

```python
from video.editor import TimelineClip
from pathlib import Path

timeline = [
    TimelineClip(
        clip_path=Path('clip1.mp4'),
        start_time=0.0,
        duration=3.0
    ),
    TimelineClip(
        clip_path=Path('clip2.mp4'),
        start_time=3.0,
        duration=4.0
    )
]

# Render custom timeline
generator.video_editor.assemble_timeline(timeline, Path('output.mp4'))
```

---

## Troubleshooting

### Issue: FFmpeg not found

**Solution:**
```bash
# Linux
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

### Issue: Out of memory with large clip pools

**Solution:** Reduce batch size in config:
```yaml
semantics:
  batch_size: 8  # Lower from 32
```

### Issue: Slow processing on CPU

**Solution:** Use GPU:
```yaml
semantics:
  device: "cuda"  # Ensure CUDA-capable GPU installed
```

### Issue: Black bars in output

**Solution:** Enable no_black_bars:
```yaml
output:
  no_black_bars: true
```

### Issue: Poor clip matches

**Solution:** Adjust semantic threshold:
```yaml
clip_selection:
  semantic_threshold: 0.5  # Lower for more permissive matching
  diversity_factor: 0.5    # Increase diversity
```

### Issue: Video codec not supported

**Solution:** Use different codec:
```yaml
video:
  codec: "libx265"  # H.265 (better compression)
  # or
  codec: "libvpx-vp9"  # VP9 (open source)
```

---

## Performance Tips

1. **GPU Processing**: Use CUDA for 5-10x faster embeddings
2. **Batch Size**: Increase for faster processing if VRAM available
3. **Resolution**: 1080p is balance of quality vs speed
4. **Codec Preset**: Use "fast" for drafts, "slow" for final
5. **Clip Cache**: Re-uses cached metadata on subsequent runs

---

## Output Versioning

Automatically prevents overwrites:

```
output/
├── musicvideo_v001.mp4  # First generation
├── musicvideo_v002.mp4  # Rerun with different config
├── musicvideo_v003.mp4  # Another generation
└── ...
```

Disable with:
```yaml
output:
  auto_versioning: false
```

---

## CLI Options

```bash
python main.py --help

Options:
  --song TEXT              MP3 file path [required]
  --clips-dir TEXT         Clip directory path [required]
  --output TEXT            Output directory [default: ./output]
  --config TEXT            Config YAML file
  --fps INTEGER            Output frames per second [default: 30]
  --resolution TEXT        Output resolution [default: 1920x1080]
  --verbose                Enable verbose output
```

---

## Support & Contributing

For issues or contributions:
- 📝 Create an issue on GitHub
- 🔧 Submit pull requests
- 💡 Suggest features

---

**Made with ❤️ by the WE.ED Team**
