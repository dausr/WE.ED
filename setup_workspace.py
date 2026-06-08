import os
import json

# Definiert die komplette Projektstruktur und den initialen Inhalt jeder Datei
PROJECT_STRUCTURE = {
    # --- Root Files ---
    "main.py": """from weedit_engine import WEEDitEngine

if __name__ == "__main__":
    print("🚀 Starting WE.ED.IT AI Music Video Director")
    print("Music directs. AI cuts. You create.")
    engine = WEEDitEngine()
    engine.run_full_auto()
    print("✅ All done! Check ./done folder")
""",
    "weedit_engine.py": """import os
import json
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Placeholder imports for the modular architecture
# from audio_analyzer import AudioAnalyzer
# from clip_analyzer import ClipAnalyzer
# from matcher import ClipMatcher
# from renderer import Renderer

class WEEDitEngine:
    def __init__(self):
        self.config = self.load_config()
        self.hardware = self.detect_hardware()
        self.write_nfo()
        print(f"[WE.ED.IT] Initialized - Hardware Mode: {self.hardware['mode']}")

    def load_config(self):
        if os.path.exists("config.json"):
            with open("config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def detect_hardware(self):
        profile = {"gpu": False, "vram_gb": 0.0, "mode": "full"}
        try:
            if torch.cuda.is_available():
                profile["gpu"] = True
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                profile["vram_gb"] = round(vram, 2)
                if vram < 6:
                    profile["mode"] = "low_cost"
        except Exception:
            pass
        return profile

    def write_nfo(self):
        os.makedirs("NFOs", exist_ok=True)
        with open("NFOs/hardware_profile.json", "w", encoding="utf-8") as f:
            json.dump(self.hardware, f, indent=4)

    def scan_clips_recursive(self):
        valid_ext = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}
        clips = []
        for pool in self.config.get("clip_pools", ["./clips"]):
            if os.path.exists(pool):
                print(f"[WE.ED.IT] Scanning {pool} ...")
                for file in Path(pool).rglob("*"):
                    if file.suffix.lower() in valid_ext:
                        clips.append(str(file))
        return list(set(clips))

    def run_full_auto(self):
        os.makedirs(self.config.get("output_folder", "./done"), exist_ok=True)
        print("[WE.ED.IT] Pipeline ready. Waiting for module implementation.")
""",
    "audio_analyzer.py": """import librosa
import numpy as np
import json
from mutagen.mp3 import MP3
from pathlib import Path
from datetime import datetime

class AudioAnalyzer:
    def analyze(self, mp3_path: str, generate_plot: bool = True) -> dict:
        \"\"\"Advanced audio analysis with SuperFlux-inspired onset detection\"\"\"
        print(f"[AudioAnalyzer] Analyzing {Path(mp3_path).name}...")
        # TODO: Implement full SuperFlux + HPSS logic from knowledge base
        return {"bpm": 128.0, "duration": 180.0, "status": "stub"}
""",
    "clip_analyzer.py": """import cv2
import numpy as np
from pathlib import Path

class ClipAnalyzer:
    def __init__(self):
        self.farneback_presets = {
            "default": {"pyr_scale": 0.5, "levels": 3, "winsize": 15, "iterations": 3, "poly_n": 5, "poly_sigma": 1.2, "flags": 0},
            "fast": {"pyr_scale": 0.8, "levels": 2, "winsize": 11, "iterations": 2, "poly_n": 5, "poly_sigma": 1.1, "flags": 0},
            "high_quality": {"pyr_scale": 0.5, "levels": 5, "winsize": 21, "iterations": 5, "poly_n": 7, "poly_sigma": 1.5, "flags": cv2.OPTFLOW_FARNEBACK_GAUSSIAN}
        }

    def analyze(self, clip_path: str):
        \"\"\"Advanced Optical Flow analysis with adaptive Farneback tuning\"\"\"
        print(f"[ClipAnalyzer] Analyzing {Path(clip_path).name}...")
        # TODO: Implement Farneback Optical Flow scoring logic
        return {"motion_score": 0.5, "dynamic_variance": 0.5, "status": "stub"}
""",
    "matcher.py": """from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ClipMatcher:
    def __init__(self):
        self.clip_db = {}

    def add_clip(self, path: str, vector):
        self.clip_db[path] = np.array(vector)

    def match(self, song_vector, gender_preference=None, top_k=20):
        \"\"\"Bidirectional Tinder Match: Song needs ↔ Clip offers\"\"\"
        # TODO: Implement cosine similarity + gender vector boosting
        return []
""",
    "renderer.py": """from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

class Renderer:
    def render(self, song_path, clip_paths, song_analysis, output_path):
        \"\"\"Beat-synced rendering with dynamic zoom, brightness pulse, and motion blur\"\"\"
        print(f"[Renderer] Building video with {len(clip_paths)} clips...")
        # TODO: Implement beat-synced cutting and VFX application
        print("[Renderer] Render stub complete.")
""",
    "requirements.txt": """librosa>=0.10.0
numpy
torch torchvision torchaudio
moviepy
mutagen
ffmpeg-python
pillow
scikit-learn
tqdm
opencv-python
matplotlib
pandas
""",
    "config.json": json.dumps({
        "clip_pools": [
            "H:\\\\repos\\\\NFOs\\\\Clips",
            "H:\\\\raw_vidz\\\\grok\\\\_16ZU9",
            "H:\\\\raw_vidz\\\\grok\\\\_new___",
            "H:\\\\raw_vidz\\\\grok\\\\1zu1",
            "H:\\\\raw_vidz\\\\grok\\\\9zu16",
            "./clips"
        ],
        "mp3_folder": "./mp3s",
        "output_folder": "./done",
        "max_clips_per_video": 50,
        "low_vram_mode": False,
        "gender_boost": True
    }, indent=4),
    "run_weedit.bat": """@echo off
title WE.ED.IT - AI Beat-Sync Video Director
echo =============================================
echo        WE.ED.IT AI Director v1.0
echo  Music directs. AI cuts. You create.
echo =============================================
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>nul
python -m pip install -r requirements.txt --quiet
python main.py
echo.
echo =============================================
echo Process finished. Videos saved in ./done
pause
""",
    "README.md": """# WE.ED.IT – AI Art Director & Intelligent Beat-Sync Video Editor
**"Music directs. AI cuts. You create."**

Fully local, automatic music video generator. Drop one MP3 → get a professional beat-synced video.

## Features
- Pro-level audio analysis (SuperFlux BPM, beats, drops, structure)
- Deep clip understanding (Farneback Optical Flow: shot scale, motion, lighting, gender, mood)
- Smart Vector Matchmaking ("Tinder for Media")
- Automatic professional cutting + visual storytelling (zoom, brightness pulse, motion blur)
- Realistic Viral Scoring with improvement tips
- Hardware-aware (works on old GPUs via low_cost_mode)
- 100% local & private

## Quick Start
1. `pip install -r requirements.txt`
2. Put MP3s in `mp3s/`
3. Put clips in `clips/` or configure paths in `config.json`
4. Run `run_weedit.bat` (Windows) or `python main.py`
5. Enjoy video in `done/`
""",

    # --- Directories (Keys ending with / are treated as directories) ---
    "agents/": "",
    "agents/__init__.py": "# Multi-Agent Layer Architecture",
    "agents/labatneda.py": "# Meta-Tag Pre-Understander & Far-Seer\n\nclass LabatnedaAgent:\n    pass",
    "agents/mucaufqua.py": "# Poly-VFXler & Digital Pyrotechnician\n\nclass MucaufquaAgent:\n    pass",
    "agents/director.py": "# State-of-the-Art Director's Cutter\n\nclass DirectorAgent:\n    pass",
    "agents/reviewer.py": "# Quality gate + Viral Scoring\n\nclass ReviewerAgent:\n    pass",
    "NFOs/": "",
    "mp3s/": "",
    "clips/": "",
    "done/": "",
    "db/": "",
    "db/schnittmuster_db.json": json.dumps({"metadata": {"version": "1.0", "total_tracks": 0}, "tracks": []}, indent=4),
    "db/clip_vault_db.json": json.dumps({"metadata": {"version": "1.0", "total_clips": 0}, "clips": []}, indent=4),
}

def setup_workspace():
    print("🚀 Setting up WE.ED.IT workspace...")
    
    for filepath, content in PROJECT_STRUCTURE.items():
        # Create directories if the path implies a folder or is explicitly a folder
        if filepath.endswith("/") or not os.path.splitext(filepath)[1]:
            os.makedirs(filepath, exist_ok=True)
            print(f"📁 Created directory: {filepath}")
        else:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(filepath)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            
            # Write file if it doesn't exist (to prevent overwriting user work)
            if not os.path.exists(filepath):
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"📄 Created file: {filepath}")
            else:
                print(f"⚠️  Skipped (already exists): {filepath}")

    print("\n✅ WE.ED.IT Workspace setup complete!")
    print("💡 Next step: Run 'python main.py' or double-click 'run_weedit.bat'")

if __name__ == "__main__":
    setup_workspace()