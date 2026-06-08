# 🎬 WE.ED - AI-Powered Beat-Sync Music Video Creator

**Vollautomatisierter Musikvideo-Generator mit semantischer Clip-Auswahl und professionellen Übergängen**

## 🎯 Features

- ✅ **Automatische MP3-Analyse** - Beat Detection, Tempo, Genre
- ✅ **Semantische Clip-Auswahl** - AI-gesteuerte Clip-Pool-Matching
- ✅ **Perfect Beat Sync** - Schnitte präzise zum Rhythmus
- ✅ **Professionelle Übergänge** - Blend Effects, Bass-Wobble, Dynamic Zoom
- ✅ **Smart Versioning** - Automatisches Versioning ohne Überschreiben
- ✅ **16:9 Standard** - Aspect Ratio Preservation
- ✅ **Clip-Wiederholung vermeiden** - Intelligente Rotation

## 🏗️ Architektur

```
WE.ED/
├── audio/              # Audio-Analyse Layer
│   ├── analyzer.py     # MP3-Analyse, Beat Detection, BPM
│   ├── metadata.py     # MP3-Tag Extraktion (Artist, Genre, etc.)
│   └── beat_sync.py    # Beat-Detection & Tempo Mapping
├── clips/              # Clip-Management Layer
│   ├── pool.py         # Clip-Pool Management
│   ├── selector.py     # Semantische Clip-Auswahl (Vector Logic)
│   └── preprocessor.py # Clip-Normalisierung & Metadaten
├── video/              # Video-Processing Layer
│   ├── editor.py       # Core FFmpeg Video-Editing
│   ├── transitions.py  # Blend Effects, Wobble, Zoom
│   ├── renderer.py     # Output Rendering Pipeline
│   └── codec.py        # Codec & Format Management
├── semantics/          # AI/Semantic Layer
│   ├── embeddings.py   # Video/Audio Embedding Models
│   ├── matching.py     # Semantic Matching Logic
│   └── context.py      # Song Context Analysis
├── config/             # Konfiguration
│   ├── settings.py     # Global Settings
│   └── defaults.yaml   # Default Parameter
├── pipeline.py         # Main Orchestration Pipeline
├── main.py             # CLI Entry Point
├── requirements.txt    # Dependencies
└── examples/           # Example Configurations
```

## 🚀 Quick Start

```bash
# Installation
pip install -r requirements.txt

# Musikvideo erstellen
python main.py --song song.mp3 --clips-dir ./clips --output videos/

# Mit Custom-Konfiguration
python main.py --song song.mp3 --config config.yaml
```

## 📋 Workflow

1. **Audio Analyse** → BPM, Beats, Genre aus MP3-Tags
2. **Semantische Analyse** → Song-Kontext Embedding
3. **Clip-Auswahl** → Semantisch passende Clips aus Pool
4. **Beat-Sync Schnitt** → Schnitte zu Beat-Positionen
5. **Übergänge** → Blend Effects, Bass-Wobble-Zoom
6. **Rendering** → 16:9 Output mit Smart Versioning

## 🎨 Transition Effects

- **Blend Transitions** - Smooth Cross-Fade
- **Bass-Wobble Zoom** - Dynamisches Zoom zur Bass-Intensität
- **Stutter Effects** - Beat-synchronized Repeat Freeze Frames
- **Color Grading** - Semantic Context based Color

## 📝 Lizenz

MIT
