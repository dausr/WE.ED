import os
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
