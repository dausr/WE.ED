# WE.ED Framework - Installation & Setup Guide

## System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 10GB disk space for clip cache
- FFmpeg/FFprobe

### Recommended
- Python 3.10+
- 16GB+ RAM
- NVIDIA GPU with CUDA support (10GB+ VRAM)
- 50GB+ SSD storage

---

## Step-by-Step Installation

### 1. Install FFmpeg

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

Verify:
```bash
ffmpeg -version
```

### 2. Clone/Download Repository

```bash
git clone https://github.com/dausr/WE.ED.git
cd WE.ED
```

### 3. Create Virtual Environment

```bash
# Create venv
python -m venv venv

# Activate
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Optional: GPU Support (CUDA)

**For NVIDIA GPU:**
```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For different CUDA versions, see: https://pytorch.org/get-started/locally/
```

Verify:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 6. Download CLIP Model (First Run)

```bash
# This will download ~350MB model
python -c "import clip; clip.load('clip-vit-base-patch32', device='cuda')"
```

---

## Project Setup

### Create Directory Structure

```bash
mkdir -p sounds clips output processed_videos
```

### Verify Installation

```bash
# Test all components
python -c "
from audio.analyzer import AudioAnalyzer
from clips.pool import ClipPool
from semantics.analyzer import SongContextAnalyzer
from video.transitions import TransitionEffects
print('✓ All imports successful')
"
```

---

## Configuration

### Default Configuration

Copy default config:
```bash
cp config/defaults.yaml config/my_config.yaml
```

### Edit Configuration

```yaml
# config/my_config.yaml
video:
  frame_rate: 30      # Adjust quality
  bitrate: "8000k"    # Video bitrate

semantics:
  device: "cuda"      # Use GPU
  batch_size: 32      # Increase if VRAM available
```

---

## First Run

### 1. Add Content

```bash
# Add MP3 file
cp /path/to/your/song.mp3 sounds/

# Add video clips (any format: .mp4, .mkv, .avi, etc.)
cp /path/to/clips/*.mp4 clips/
```

### 2. Generate Video

```bash
python main.py \
  --song sounds/song.mp3 \
  --clips-dir clips \
  --output output/ \
  --verbose
```

### 3. Output

Your video will be at: `output/musicvideo_v001.mp4`

---

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError`

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### FFmpeg Not Found

**Problem:** `ffmpeg: command not found`

**Solution:**
```bash
# Add FFmpeg to PATH
# Or install it using package manager (see above)

# Windows: Add C:\ffmpeg\bin to System PATH
```

### CUDA Not Available

**Problem:** `torch.cuda.is_available() returns False`

**Solution:**
```bash
# Check NVIDIA GPU
nvidia-smi

# Reinstall PyTorch for your CUDA version
# Visit: https://pytorch.org/get-started/locally/
```

### Out of Memory

**Problem:** `CUDA out of memory`

**Solution:**
```yaml
# config/my_config.yaml
semantics:
  batch_size: 8    # Reduce from 32
  device: "cpu"    # Fallback to CPU
```

### Slow Processing

**Problem:** Takes too long

**Solution:**
```yaml
video:
  preset: "fast"   # Instead of "medium"

semantics:
  device: "cuda"   # Use GPU
  batch_size: 64   # Increase if possible
```

---

## Environment Variables (Optional)

```bash
# Set custom clip directory
export WE_ED_CLIPS_DIR="/path/to/clips"

# Set custom output directory  
export WE_ED_OUTPUT_DIR="/path/to/output"

# Force CPU (no GPU)
export WE_ED_DEVICE="cpu"
```

---

## Docker Setup (Optional)

### Build Image

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "main.py"]
```

### Run in Docker

```bash
# Build
docker build -t we-ed .

# Run
docker run -v $(pwd)/sounds:/app/sounds \
           -v $(pwd)/clips:/app/clips \
           -v $(pwd)/output:/app/output \
           we-ed --song sounds/song.mp3 \
                 --clips-dir clips \
                 --output output/
```

---

## Next Steps

1. ✅ Installation complete
2. 📖 Read [USAGE_GUIDE.md](USAGE_GUIDE.md)
3. 🎬 Try basic example
4. 🔧 Customize config
5. 📚 Explore advanced features

---

## Support

- 📝 Issues: [GitHub Issues](https://github.com/dausr/WE.ED/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/dausr/WE.ED/discussions)
- 📧 Contact: [Email Support]
