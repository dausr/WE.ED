from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import eyed3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MUSIC_DIR = PROJECT_DIR / "test"
DEFAULT_CLIPS_DIR = Path(r"D:\Oidasheim\NFOs\Clips")
DEFAULT_CACHE_PATH = PROJECT_DIR / "outputs" / "clip_cache.json"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


@dataclass(frozen=True)
class ClipInfo:
    path: Path
    duration: float
    tags: str
    shot_scale: str
    camera_movement: str
    hex_spectrum: str


def run(command: list[str], quiet: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        if not quiet:
            print(result.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(command)}")
    return result


def require_tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} was not found on PATH.")
    return found


def media_duration(path: Path, ffprobe: str) -> float:
    result = run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        quiet=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def list_media(folder: Path, extensions: set[str], recursive: bool = True) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in extensions
    )


def choose_music(music_dir: Path, requested: str | None) -> Path:
    tracks = list_media(music_dir, AUDIO_EXTENSIONS)
    if not tracks:
        raise RuntimeError(f"No music files found in {music_dir}")
    if requested:
        requested_path = Path(requested)
        if requested_path.exists():
            return requested_path
        matches = [track for track in tracks if requested.lower() in track.name.lower()]
        if matches:
            return matches[0]
        raise RuntimeError(f"No matching music file found for: {requested}")
    return tracks[0]


def decode_audio_bytes(music_path: Path, duration: float, sample_rate: int = 22050) -> np.ndarray:
    ffmpeg = require_tool("ffmpeg")
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(music_path),
            "-t",
            f"{duration:.3f}",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return np.frombuffer(result.stdout, dtype=np.float32)


def analyze_beats(music_path: Path, target_duration: float, samples: np.ndarray | None = None) -> tuple[float, list[float]]:
    sample_rate = 22050
    hop = 512
    if samples is None:
        samples = decode_audio_bytes(music_path, target_duration, sample_rate=sample_rate)
    if samples.size < sample_rate:
        return 120.0, [0.0, round(target_duration, 3)]

    frame_count = 1 + max(0, (samples.size - hop) // hop)
    trimmed = samples[: frame_count * hop]
    frames = trimmed.reshape(frame_count, hop)
    energy = np.sqrt(np.mean(frames * frames, axis=1))
    if energy.max() > 0:
        energy = energy / energy.max()

    onset = np.maximum(0.0, np.diff(energy, prepend=energy[0]))
    if onset.size >= 5:
        onset = np.convolve(onset, np.ones(5) / 5, mode="same")

    tempo = estimate_tempo(onset, sample_rate, hop)
    min_distance = max(1, int((60.0 / max(tempo, 1.0)) * 0.45 * sample_rate / hop))
    threshold = float(np.percentile(onset, 78))
    peak_indexes = pick_peaks(onset, threshold, min_distance)
    peak_times = [index * hop / sample_rate for index in peak_indexes]

    beats = [0.0]
    beats.extend(time for time in peak_times if 0.15 < time < target_duration - 0.1)
    if len(beats) < max(4, target_duration / 2.0):
        beat_step = 60.0 / tempo
        offset = beats[1] if len(beats) > 1 else 0.0
        beats = [0.0] + [
            time for time in frange(offset, target_duration, beat_step)
            if 0.15 < time < target_duration - 0.1
        ]
    beats.append(float(target_duration))
    return tempo, sorted(set(round(time, 3) for time in beats))


def estimate_tempo(onset: np.ndarray, sample_rate: int, hop: int) -> float:
    centered = onset - np.mean(onset)
    if np.allclose(centered, 0):
        return 120.0
    corr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    min_bpm, max_bpm = 80.0, 180.0
    min_lag = max(1, int((60.0 / max_bpm) * sample_rate / hop))
    max_lag = min(len(corr) - 1, int((60.0 / min_bpm) * sample_rate / hop))
    if max_lag <= min_lag:
        return 120.0
    lag = int(np.argmax(corr[min_lag:max_lag]) + min_lag)
    return float(60.0 * sample_rate / (lag * hop))


def pick_peaks(values: np.ndarray, threshold: float, min_distance: int) -> list[int]:
    candidates = [
        index
        for index in range(1, len(values) - 1)
        if values[index] >= threshold
        and values[index] >= values[index - 1]
        and values[index] >= values[index + 1]
    ]
    candidates.sort(key=lambda index: values[index], reverse=True)
    selected: list[int] = []
    blocked = np.zeros(len(values), dtype=bool)
    for index in candidates:
        if blocked[index]:
            continue
        selected.append(index)
        low = max(0, index - min_distance)
        high = min(len(values), index + min_distance + 1)
        blocked[low:high] = True
    return sorted(selected)


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current < stop:
        values.append(current)
        current += step
    return values


def build_cut_points(beats: list[float], cut_every: int, min_segment: float) -> list[float]:
    cut_every = max(1, cut_every)
    selected = [beats[0]]
    for index in range(cut_every, len(beats) - 1, cut_every):
        if beats[index] - selected[-1] >= min_segment:
            selected.append(beats[index])
    if beats[-1] - selected[-1] >= min_segment / 2:
        selected.append(beats[-1])
    else:
        selected[-1] = beats[-1]
    return selected


def clean_text(text: str) -> str:
    # Replace non-alphanumeric characters and underscores with space, lowercase
    cleaned = re.sub(r'[\W_]+', ' ', text)
    return cleaned.strip().lower()


def stringify_metadata(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(stringify_metadata(item) for item in value)
    if isinstance(value, dict):
        return " ".join(stringify_metadata(item) for item in value.values())
    return str(value)


def safe_filename(value: str, fallback: str = "track") -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value).strip(" ._")
    return cleaned[:120] or fallback


def escape_concat_path(path: Path) -> str:
    return path.as_posix().replace("'", "'\\''")


def load_clip_cache(cache_path: Path, refresh: bool) -> dict[str, dict[str, object]]:
    if refresh or not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    clips = data.get("clips", {})
    return clips if isinstance(clips, dict) else {}


def save_clip_cache(cache_path: Path, clips: dict[str, dict[str, object]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "clips": clips}
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cached_duration(path: Path, ffprobe: str, cache: dict[str, dict[str, object]]) -> float:
    key = str(path.resolve())
    stat = path.stat()
    cached = cache.get(key)
    if (
        cached
        and cached.get("size") == stat.st_size
        and cached.get("mtime") == stat.st_mtime
        and cached.get("duration") is not None
    ):
        return float(cached["duration"])

    duration = media_duration(path, ffprobe)
    entry = dict(cached) if isinstance(cached, dict) else {}
    entry.update({"size": stat.st_size, "mtime": stat.st_mtime, "duration": duration})
    cache[key] = entry
    return duration


def cached_metadata(path: Path, cache: dict[str, dict[str, object]]) -> tuple[str, str, str, str]:
    key = str(path.resolve())
    stat = path.stat()
    nfo_path = path.with_suffix(".nfo")
    nfo_mtime = nfo_path.stat().st_mtime if nfo_path.exists() else None
    cached = cache.get(key)
    if (
        cached
        and cached.get("size") == stat.st_size
        and cached.get("mtime") == stat.st_mtime
        and cached.get("nfo_mtime") == nfo_mtime
        and "tags" in cached
    ):
        return (
            stringify_metadata(cached.get("tags", "")),
            stringify_metadata(cached.get("shot_scale", "")),
            stringify_metadata(cached.get("camera_movement", "")),
            stringify_metadata(cached.get("hex_spectrum", "")),
        )

    tags = ""
    shot_scale = ""
    camera_movement = ""
    hex_spectrum = ""
    if nfo_path.exists():
        try:
            with open(nfo_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                tags = stringify_metadata(meta.get("tags", ""))
                shot_scale = stringify_metadata(meta.get("shot_scale", ""))
                camera_movement = stringify_metadata(meta.get("camera_movement", ""))
                hex_spectrum = stringify_metadata(meta.get("hex_spectrum", ""))
        except Exception as e:
            print(f"[Warning] Error reading metadata for {path.name}: {e}")

    entry = dict(cached) if isinstance(cached, dict) else {}
    entry.update(
        {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "nfo_mtime": nfo_mtime,
            "tags": tags,
            "shot_scale": shot_scale,
            "camera_movement": camera_movement,
            "hex_spectrum": hex_spectrum,
        }
    )
    cache[key] = entry
    return tags, shot_scale, camera_movement, hex_spectrum


def get_segment_energy(samples: np.ndarray, sample_rate: int, start: float, end: float) -> float:
    start_idx = int(start * sample_rate)
    end_idx = int(end * sample_rate)
    segment_samples = samples[start_idx:end_idx]
    if segment_samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(segment_samples * segment_samples)))


def build_dynamic_cut_points(beats: list[float], samples: np.ndarray, sample_rate: int, min_segment: float) -> list[float]:
    beat_energies = []
    for start, end in zip(beats, beats[1:]):
        energy = get_segment_energy(samples, sample_rate, start, end)
        beat_energies.append(energy)
    
    # Smooth energy over a running window of size 5
    smoothed_energies = []
    w = 5
    for i in range(len(beat_energies)):
        start_idx = max(0, i - w // 2)
        end_idx = min(len(beat_energies), i + w // 2 + 1)
        smoothed_energies.append(float(np.mean(beat_energies[start_idx:end_idx])))
        
    if smoothed_energies:
        max_se = max(smoothed_energies)
        min_se = min(smoothed_energies)
        range_se = max_se - min_se
    else:
        max_se, min_se, range_se = 1.0, 0.0, 1.0
        
    cut_points = [beats[0]]
    i = 0
    while i < len(beats) - 1:
        energy = smoothed_energies[i] if i < len(smoothed_energies) else 0.0
        norm_energy = (energy - min_se) / range_se if range_se > 0 else 0.5
        
        # Decide cutting grid dynamically based on energy density
        if norm_energy >= 0.7:
            step = 1  # Cut every beat in high energy sections
        elif norm_energy >= 0.3:
            step = 2  # Cut every 2 beats in medium energy sections
        else:
            step = 4  # Cut every 4 beats in low energy sections
            
        next_i = min(i + step, len(beats) - 1)
        if beats[next_i] - cut_points[-1] >= min_segment:
            cut_points.append(beats[next_i])
            i = next_i
        else:
            i += 1
            
    if beats[-1] - cut_points[-1] >= min_segment / 2:
        cut_points.append(beats[-1])
    else:
        cut_points[-1] = beats[-1]
        
    return cut_points


def select_clip(
    clips: list[ClipInfo],
    segment_duration: float,
    segment_energy: float,
    min_se: float,
    max_se: float,
    similarities: np.ndarray,
    used_clips: list[Path],
    cooldown_limit: int,
    rng: random.Random
) -> ClipInfo:
    range_se = max_se - min_se
    norm_energy = (segment_energy - min_se) / range_se if range_se > 0 else 0.5

    # Filter by duration (add buffer)
    valid_clips = [c for c in clips if c.duration >= segment_duration + 0.15]
    if not valid_clips:
        valid_clips = clips

    # Filter out recently used clips for repetition avoidance
    available_clips = [c for c in valid_clips if c.path not in used_clips]
    if not available_clips:
        # Prune the history to release the oldest clips
        for _ in range(len(used_clips) // 2):
            if used_clips:
                used_clips.pop(0)
        available_clips = [c for c in valid_clips if c.path not in used_clips]
        if not available_clips:
            available_clips = valid_clips

    best_clip = None
    best_score = -1.0

    for clip in available_clips:
        idx = clips.index(clip)
        sim = similarities[idx]

        # Calculate energy matching bonus (match camera movement and tags to energy)
        energy_bonus = 0.0
        movement = clip.camera_movement.lower()
        tags = clip.tags.lower()

        is_fast_clip = "fast" in movement or "fast" in tags or "action" in tags or "epic" in tags
        is_slow_clip = "slow" in movement or "static" in movement or "slow" in tags or "calm" in tags or "moody" in tags

        if norm_energy >= 0.6 and is_fast_clip:
            energy_bonus = 0.3
        elif norm_energy <= 0.4 and is_slow_clip:
            energy_bonus = 0.3

        # Tiny random jitter (0.0 to 0.05) to ensure diversity
        jitter = rng.uniform(0.0, 0.05)

        score = sim + energy_bonus + jitter

        if score > best_score:
            best_score = score
            best_clip = clip

    if not best_clip:
        best_clip = rng.choice(available_clips)

    # Track usage and pop oldest to maintain size under limit
    used_clips.append(best_clip.path)
    if len(used_clips) > cooldown_limit:
        used_clips.pop(0)

    return best_clip


def probe_clips(clips_dir: Path, ffprobe: str, cache_path: Path, refresh_cache: bool) -> list[ClipInfo]:
    clips: list[ClipInfo] = []
    cache = load_clip_cache(cache_path, refresh_cache)
    changed = False
    for path in list_media(clips_dir, VIDEO_EXTENSIONS):
        try:
            before = dict(cache.get(str(path.resolve()), {}))
            duration = cached_duration(path, ffprobe, cache)
            tags, shot_scale, camera_movement, hex_spectrum = cached_metadata(path, cache)
            if before != cache.get(str(path.resolve()), {}):
                changed = True
        except Exception:
            continue
        if duration > 0.5:
            clips.append(ClipInfo(
                path=path,
                duration=duration,
                tags=tags,
                shot_scale=shot_scale,
                camera_movement=camera_movement,
                hex_spectrum=hex_spectrum
            ))
    if not clips:
        raise RuntimeError(f"No usable clips found in {clips_dir}")
    if changed:
        save_clip_cache(cache_path, cache)
    return clips


def render_segment(
    ffmpeg: str,
    clip: ClipInfo,
    segment_duration: float,
    output: Path,
    rng: random.Random,
    orientation: str,
    quality: str,
) -> None:
    start = 0.0
    if clip.duration > segment_duration + 0.35:
        start = rng.uniform(0.0, clip.duration - segment_duration - 0.15)

    if orientation == "landscape":
        size = "1920:1080"
    elif orientation == "square":
        size = "1080:1080"
    else:
        size = "1080:1920"

    vf = (
        f"scale={size}:force_original_aspect_ratio=increase,"
        f"crop={size},fps=30,setsar=1,setpts=PTS-STARTPTS"
    )
    run(
        [
            ffmpeg,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(clip.path),
            "-t",
            f"{segment_duration:.3f}",
            "-an",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            quality,
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )


def mux_segments(
    ffmpeg: str,
    segment_paths: list[Path],
    music_path: Path,
    output_path: Path,
    work_dir: Path,
    target_duration: float,
) -> None:
    concat_file = work_dir / "concat.txt"
    concat_file.write_text(
        "".join(f"file '{escape_concat_path(path)}'\n" for path in segment_paths),
        encoding="utf-8",
    )
    silent_video = work_dir / "silent_video.mp4"
    run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(silent_video),
        ]
    )
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(silent_video),
            "-i",
            str(music_path),
            "-t",
            f"{target_duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ]
    )


def process_track(music_path: Path, args: argparse.Namespace) -> Path:
    ffmpeg = require_tool("ffmpeg")
    ffprobe = require_tool("ffprobe")
    clips_dir = Path(args.clips_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    music_duration = media_duration(music_path, ffprobe)
    target_duration = min(music_duration, args.duration) if args.duration else music_duration
    
    # 1. Beat detection and decoding
    sample_rate = 22050
    print(f"\n[Audio] Loading and analyzing beats for: {music_path.name}")
    samples = decode_audio_bytes(music_path, target_duration, sample_rate=sample_rate)
    tempo, beats = analyze_beats(music_path, target_duration, samples=samples)
    
    # 2. Dynamic cut points based on energy
    if args.dynamic_pacing:
        cut_points = build_dynamic_cut_points(beats, samples, sample_rate, args.min_segment)
    else:
        cut_points = build_cut_points(beats, args.cut_every, args.min_segment)
        
    clips = probe_clips(
        clips_dir=clips_dir,
        ffprobe=ffprobe,
        cache_path=Path(args.clip_cache),
        refresh_cache=args.refresh_clip_cache,
    )
    
    # 3. Extract Song Context metadata
    print("[Metadata] Extracting song metadata context...")
    title = music_path.stem
    artist = ""
    genre = ""
    lyrics = ""
    try:
        audio = eyed3.load(str(music_path))
        if audio and audio.tag:
            title = audio.tag.title if audio.tag.title else title
            artist = audio.tag.artist if audio.tag.artist else ""
            genre = audio.tag.genre.name if audio.tag.genre else ""
            if audio.tag.lyrics:
                lyrics = " ".join(l.text for l in audio.tag.lyrics if l.text)
    except Exception as e:
        print(f"[Warning] Warning loading ID3 tags: {e}")
        
    song_query = clean_text(f"{title} {artist} {genre} {lyrics} {music_path.stem}")
    print(f"   Query keywords: {song_query}")

    # 4. TF-IDF vectorizer similarity
    print("[Matching] Computing semantic similarities...")
    corpus = [clean_text(f"{c.tags} {c.shot_scale} {c.camera_movement}") for c in clips]
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vector = vectorizer.transform([song_query])
        similarities = (tfidf_matrix * query_vector.T).toarray().flatten()
    except Exception as e:
        print(f"[Warning] Vectorizer fallback: {e}")
        similarities = np.zeros(len(clips))
        
    # 5. Segment energies
    segment_energies = [
        get_segment_energy(samples, sample_rate, start, end)
        for start, end in zip(cut_points, cut_points[1:])
    ]
    max_se = max(segment_energies) if segment_energies else 1.0
    min_se = min(segment_energies) if segment_energies else 0.0

    rng = random.Random(args.seed)
    
    # 6. Clip selection with repetition avoidance
    cooldown_limit = max(1, len(clips) // 2)
    used_clips: list[Path] = []
    
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"co_naz_{safe_filename(music_path.stem)}_{stamp}.mp4"

    print(f"Tempo: {tempo:.1f} BPM")
    print(f"Total Clips in pool: {len(clips)}")
    print(f"Total Cuts: {len(cut_points) - 1}")
    print(f"Output Path: {output_path}")

    with tempfile.TemporaryDirectory(prefix="co-naz-") as temp_name:
        work_dir = Path(temp_name)
        segment_paths: list[Path] = []
        for index, (start, end) in enumerate(zip(cut_points, cut_points[1:]), start=1):
            duration = max(0.2, end - start)
            
            # Get local segment energy
            seg_energy = segment_energies[index - 1]
            
            # Select clip using semantic vector match + energy profile + repetition avoidance
            clip = select_clip(
                clips=clips,
                segment_duration=duration,
                segment_energy=seg_energy,
                min_se=min_se,
                max_se=max_se,
                similarities=similarities,
                used_clips=used_clips,
                cooldown_limit=cooldown_limit,
                rng=rng
            )
            
            segment_path = work_dir / f"segment_{index:04d}.mp4"
            print(f"[{index:03d}/{len(cut_points) - 1:03d}] {clip.path.name} ({duration:.2f}s) [energy: {seg_energy:.4f}]")
            render_segment(
                ffmpeg=ffmpeg,
                clip=clip,
                segment_duration=duration,
                output=segment_path,
                rng=rng,
                orientation=args.orientation,
                quality=args.preset,
            )
            segment_paths.append(segment_path)
            
        print("[Muxer] Muxing all segments and audio soundtrack...")
        mux_segments(ffmpeg, segment_paths, music_path, output_path, work_dir, target_duration)

    return output_path


def render_beat_sync(args: argparse.Namespace) -> Path:
    music_dir = Path(args.music_dir)
    music_path = choose_music(music_dir, args.music)
    return process_track(music_path, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="co.naz creates a beat-synced edit from local music and video clips."
    )
    parser.add_argument("--music-dir", default=str(DEFAULT_MUSIC_DIR), help="Folder containing audio files.")
    parser.add_argument("--clips-dir", default=str(DEFAULT_CLIPS_DIR), help="Folder containing video clips.")
    parser.add_argument("--output-dir", default="outputs", help="Folder for rendered MP4 files.")
    parser.add_argument("--clip-cache", default=str(DEFAULT_CACHE_PATH), help="Clip probe cache JSON path.")
    parser.add_argument("--refresh-clip-cache", action="store_true", help="Rebuild cached clip durations and metadata.")
    parser.add_argument("--music", help="Music filename, partial name, or full path.")
    parser.add_argument("--duration", type=float, help="Limit output length in seconds.")
    parser.add_argument("--cut-every", type=int, default=2, help="Cut every N detected beats.")
    parser.add_argument("--min-segment", type=float, default=0.45, help="Smallest allowed cut segment in seconds.")
    parser.add_argument(
        "--orientation",
        choices=["portrait", "landscape", "square"],
        default="portrait",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable clip choices.")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle songs before batch rendering.")
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle", help="Render songs alphabetically.")
    parser.add_argument("--dynamic-pacing", action="store_true", default=True)
    parser.add_argument("--no-dynamic-pacing", action="store_false", dest="dynamic_pacing")
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be greater than 0.")
    if args.cut_every < 1:
        parser.error("--cut-every must be at least 1.")
    if args.min_segment <= 0:
        parser.error("--min-segment must be greater than 0.")
    return args


def main() -> int:
    try:
        args = parse_args()
        music_dir = Path(args.music_dir)
        
        # Determine list of tracks to process
        tracks: list[Path] = []
        if args.music:
            requested_path = Path(args.music)
            if requested_path.exists() and requested_path.is_file():
                tracks.append(requested_path)
            else:
                # Search by partial filename
                all_tracks = list_media(music_dir, AUDIO_EXTENSIONS)
                matches = [track for track in all_tracks if args.music.lower() in track.name.lower()]
                if matches:
                    tracks.append(matches[0])
                else:
                    print(f"[Error] No matching music file found for: {args.music}", file=sys.stderr)
                    return 1
        else:
            # Batch process all tracks in music_dir
            tracks = list_media(music_dir, AUDIO_EXTENSIONS)
            if not tracks:
                print(f"[Error] No music files found in directory: {music_dir}", file=sys.stderr)
                return 1
            if args.shuffle:
                random.Random(args.seed).shuffle(tracks)
                
        print(f"[Pipeline] Found {len(tracks)} song(s) to process.")
        for i, track in enumerate(tracks, 1):
            print(f"\n==================================================")
            print(f"[Song] Processing Track [{i}/{len(tracks)}]: {track.name}")
            print(f"==================================================")
            output_path = process_track(track, args)
            print(f"[Done] Finished: {output_path}")
            
    except Exception as exc:
        print(f"[Error] co.naz failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
