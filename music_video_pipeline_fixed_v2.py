#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1-Click-1-File Musikvideo-Pipeline (Fixed Version v2)
Automatisches Matching von Songs und Clips basierend auf semantischen Eigenschaften.
Kompatibel mit Python 3.14, ohne torchcodec/sentence-transformers, und mit robuster Fehlerbehandlung.
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from eyed3 import load
import librosa
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

# --- 1. Abhängigkeiten prüfen und installieren ---
def install_dependencies():
    required_packages = [
        "eyed3", "librosa", "scikit-learn",
        "opencv-python", "pillow", "pandas", "numpy", "moviepy"
    ]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"🔧 Installiere {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_dependencies()

# --- 2. Konfiguration ---
SOUND_DIR = "D:/Oidasheim/NFOs/mp3s"  # Pfad zu MP3s
CLIP_DIR = "D:/Oidasheim/NFOs/Clips"   # Pfad zu Clips
OUTPUT_DIR = "./done"                   # Output-Ordner

# Erstelle Output-Ordner, falls nicht vorhanden
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 3. Hilfsfunktionen ---
def scan_directory(directory, follow_symlinks=True):
    """Rekursiv Dateien in einem Ordner scannen (inkl. Symlinks), aber .lnk-Dateien ignorieren."""
    for root, dirs, files in os.walk(directory, followlinks=follow_symlinks):
        for file in files:
            file_path = os.path.join(root, file)
            # Ignoriere .lnk-Dateien und versteckte Dateien
            if not file.endswith('.lnk') and not file.startswith('~$'):
                yield file_path

def get_dominant_color(frame):
    """Dominante Farbe eines Frames extrahieren."""
    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixels = frame.reshape(-1, 3)
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(pixels, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        return centers[0].tolist()
    except:
        return [0.0, 0.0, 0.0]  # Fallback: Schwarz

def analyze_motion(frames):
    """Bewegung zwischen Frames analysieren (vereinfacht)."""
    try:
        if len(frames) < 2:
            return 0.0
        prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        motion = 0.0
        for frame in frames[1:]:
            curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_frame, curr_frame)
            motion += np.mean(diff)
            prev_frame = curr_frame
        return motion / len(frames)
    except:
        return 0.0  # Fallback

def analyze_mood(audio_path):
    """Mood basierend auf BPM und Energie schätzen (vereinfacht)."""
    try:
        y, sr = librosa.load(audio_path, duration=30)  # Nur erste 30 Sekunden laden
        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        energy = float(np.mean(librosa.feature.rms(y=y)[0]))
        if bpm > 120 and energy > 0.5:
            return "energetic"
        elif bpm < 80:
            return "calm"
        else:
            return "neutral"
    except:
        return "neutral"  # Fallback

def analyze_energy(audio_path):
    """Energie der Audiodatei schätzen."""
    try:
        y, sr = librosa.load(audio_path, duration=30)  # Nur erste 30 Sekunden laden
        return float(np.mean(librosa.feature.rms(y=y)[0]))
    except:
        return 0.0  # Fallback

# --- 4. Metadaten-Extraktion ---
def extract_mp3_metadata(file_path):
    """Metadaten aus MP3-Datei extrahieren."""
    try:
        # Prüfe, ob die Datei eine MP3-Datei ist
        if not file_path.lower().endswith(('.mp3', '.wav', '.flac')):
            return None

        audio = load(file_path)
        bpm = 120.0  # Fallback
        energy = 0.0  # Fallback
        try:
            y, sr = librosa.load(file_path, duration=30)
            bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
            energy = float(np.mean(librosa.feature.rms(y=y)[0]))
        except:
            pass  # Fallback-Werte verwenden

        return {
            "title": audio.tag.title if audio.tag.title else Path(file_path).stem,
            "artist": audio.tag.artist if audio.tag.artist else "Unknown",
            "bpm": float(bpm),
            "lyrics": audio.tag.lyrics if audio.tag.lyrics else "",
            "genre": audio.tag.genre if audio.tag.genre else "Unknown",
            "mood": analyze_mood(file_path),
            "energy": energy,
            "file_path": file_path
        }
    except Exception as e:
        print(f"⚠️ Fehler bei {file_path}: {e}")
        return None

def extract_video_metadata(file_path):
    """Metadaten aus Video-Datei extrahieren."""
    try:
        # Prüfe, ob die Datei eine Video-Datei ist
        if not file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return None

        cap = cv2.VideoCapture(file_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            return None

        brightness = float(np.mean([np.mean(frame) for frame in frames]))
        dominant_color = get_dominant_color(frames[0])
        motion = float(analyze_motion(frames))

        return {
            "clip_id": Path(file_path).stem,
            "date": os.path.getctime(file_path),
            "brightness": brightness,
            "dominant_color": dominant_color,
            "motion": motion,
            "tags": [],  # Hier könnten KI-Tags hinzukommen
            "file_path": file_path
        }
    except Exception as e:
        print(f"⚠️ Fehler bei {file_path}: {e}")
        return None

# --- 5. Semantische Analyse (ohne sentence-transformers) ---
def generate_text_embedding(text):
    """Einfaches Text-Embedding mit TF-IDF."""
    try:
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform([text]).toarray()[0]
    except:
        return np.zeros(10)  # Fallback: 10-dimensionale Null-Vektor

def generate_numerical_embedding(features):
    """Numerische Features normalisieren und kombinieren."""
    return np.array(features)

# --- 6. Matching-Algorithmus ---
def match_song_to_clips(song, clips, song_embedding, clip_embeddings):
    """Song mit Clips matchen basierend auf Embeddings und Metadaten."""
    try:
        similarities = cosine_similarity([song_embedding], clip_embeddings)[0]
        ranked_clips = sorted(zip(clips, similarities), key=lambda x: -x[1])

        # Filterung nach Mood und Energie (optional)
        filtered_clips = []
        for clip, similarity in ranked_clips:
            if (song["mood"] == "energetic" and clip["motion"] > 50) or \
               (song["mood"] == "calm" and clip["motion"] < 30) or \
               (song["mood"] == "neutral"):
                filtered_clips.append((clip, similarity))

        return filtered_clips[:5]  # Top 5 Matches
    except:
        return []  # Fallback: Keine Matches

# --- 7. Export ---
def export_to_json(data, file_path):
    """Daten als JSON exportieren."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def export_to_csv(data, file_path):
    """Daten als CSV exportieren."""
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, encoding="utf-8")

# --- 8. Haupt-Pipeline ---
def main():
    print("🚀 Starte Musikvideo-Pipeline (Fixed Version v2)...")

    # 1. MP3-Metadaten extrahieren
    print("🎵 Extrahiere MP3-Metadaten...")
    mp3_files = list(scan_directory(SOUND_DIR))
    songs = []
    for mp3_file in mp3_files:
        song = extract_mp3_metadata(mp3_file)
        if song:
            songs.append(song)

    # 2. Video-Metadaten extrahieren
    print("🎥 Extrahiere Video-Metadaten...")
    clip_files = list(scan_directory(CLIP_DIR))
    clips = []
    for clip_file in clip_files:
        clip = extract_video_metadata(clip_file)
        if clip:
            clips.append(clip)

    if not songs or not clips:
        print("❌ Keine Songs oder Clips gefunden. Überprüfe die Pfade!")
        return

    # 3. Embeddings generieren (Text + Numerisch)
    print("🧠 Generiere Embeddings...")
    song_embeddings = []
    for song in songs:
        # Text-Embedding (Lyrics + Genre + Mood)
        text = f"{song['lyrics']} {song['genre']} {song['mood']}"
        text_embedding = generate_text_embedding(text)
        # Numerische Features (BPM, Energie)
        numerical_features = [song["bpm"], song["energy"]]
        numerical_embedding = generate_numerical_embedding(numerical_features)
        # Kombiniertes Embedding
        combined_embedding = np.concatenate([text_embedding, numerical_embedding])
        song_embeddings.append(combined_embedding)

    clip_embeddings = []
    for clip in clips:
        # Text-Embedding (Tags + Farben)
        text = f"brightness:{clip['brightness']} motion:{clip['motion']} color:{clip['dominant_color']}"
        text_embedding = generate_text_embedding(text)
        # Numerische Features (Helligkeit, Bewegung)
        numerical_features = [clip["brightness"], clip["motion"]]
        numerical_embedding = generate_numerical_embedding(numerical_features)
        # Kombiniertes Embedding
        combined_embedding = np.concatenate([text_embedding, numerical_embedding])
        clip_embeddings.append(combined_embedding)

    # 4. Matching
    print("🔍 Matche Songs mit Clips...")
    matches = []
    for song, song_emb in zip(songs, song_embeddings):
        matched_clips = match_song_to_clips(song, clips, song_emb, clip_embeddings)
        matches.append({
            "song": song,
            "matched_clips": [{"clip": clip, "similarity": float(sim)} for clip, sim in matched_clips]
        })

    # 5. Export
    print("💾 Exportiere Ergebnisse...")
    export_to_json(matches, os.path.join(OUTPUT_DIR, "matches.json"))
    export_to_csv([m["song"] for m in matches], os.path.join(OUTPUT_DIR, "songs.csv"))
    export_to_csv(
        [{"clip_id": c["clip"]["clip_id"], "similarity": c["similarity"]} for m in matches for c in m["matched_clips"]],
        os.path.join(OUTPUT_DIR, "clips.csv")
    )

    print(f"✅ Fertig! Ergebnisse in {OUTPUT_DIR} gespeichert.")

if __name__ == "__main__":
    main()