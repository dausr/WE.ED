"""Advanced Clip Information System - Standalone Intelligent Clip Asset Management"""

import os
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import numpy as np
import cv2
from tqdm import tqdm
import hashlib


@dataclass
class ClipScene:
    """Detected scene within a clip"""
    start_frame: int
    end_frame: int
    label: str
    confidence: float
    color_palette: List[Tuple[int, int, int]]
    brightness_avg: float
    contrast_avg: float


@dataclass
class ClipMotion:
    """Motion characteristics of a clip"""
    motion_score: float  # 0-1 (0=static, 1=highly dynamic)
    camera_type: str  # "static", "pan", "zoom", "handheld", "mixed"
    motion_direction: str  # "horizontal", "vertical", "circular", "none"
    fps: int
    is_slow_motion: bool
    is_timelapse: bool


@dataclass
class ClipAudio:
    """Audio characteristics (if embedded)"""
    has_audio: bool
    sample_rate: Optional[int] = None
    duration_seconds: Optional[float] = None
    audio_level_avg: Optional[float] = None
    audio_level_peak: Optional[float] = None
    is_speech_heavy: Optional[bool] = None
    estimated_genre: Optional[str] = None


@dataclass
class ClipColorAnalysis:
    """Comprehensive color analysis"""
    dominant_colors: List[Tuple[int, int, int]]
    color_palette_count: int
    saturation_avg: float
    brightness_avg: float
    contrast_avg: float
    color_temperature: str  # "warm", "neutral", "cool"
    vibrancy_score: float  # 0-1
    color_harmony: str  # "monochrome", "analogous", "complementary", "triadic"


@dataclass
class ClipContent:
    """Content/semantic analysis of clip"""
    subject_tags: List[str]  # "people", "nature", "urban", "abstract", etc.
    environment_type: str  # "indoor", "outdoor", "studio", "abstract"
    lighting_quality: str  # "natural", "artificial", "mixed", "low_key", "high_key"
    composition_style: str  # "wide", "medium", "close_up", "macro", "mixed"
    visual_style: str  # "realistic", "stylized", "animated", "mixed"
    mood_tags: List[str]  # "energetic", "calm", "dramatic", etc.
    detected_objects: Dict[str, float]  # object -> confidence
    scene_complexity: float  # 0-1 (0=simple, 1=very complex)
    estimated_duration_use: str  # "short_clip", "medium_clip", "long_clip"


@dataclass
class ClipMetadata:
    """Complete clip metadata and analysis"""
    # Basic info
    clip_id: str
    filename: str
    filepath: Path
    file_hash: str  # SHA256 for deduplication
    file_size_mb: float
    added_date: str
    
    # Technical specs
    duration_seconds: float
    width: int
    height: int
    fps: int
    total_frames: int
    codec: str
    bitrate_kbps: Optional[int]
    
    # Aspect ratio and quality
    aspect_ratio: float
    aspect_ratio_label: str  # "16:9", "4:3", "1:1", etc.
    estimated_quality: str  # "4K", "1080p", "720p", "SD"
    quality_score: float  # 0-1
    
    # Analysis results
    motion: ClipMotion
    color_analysis: ClipColorAnalysis
    content: ClipContent
    audio: ClipAudio
    scenes: List[ClipScene]
    
    # Embeddings and features
    color_histogram: Optional[np.ndarray] = None
    frame_embeddings: Optional[np.ndarray] = None  # Average of keyframe embeddings
    motion_vector: Optional[np.ndarray] = None
    semantic_tags: List[str] = field(default_factory=list)
    
    # Usage tracking
    times_used: int = 0
    last_used_date: Optional[str] = None
    user_rating: float = 0.0  # 0-5
    user_notes: str = ""
    
    # Compatibility info
    compatible_music_genres: List[str] = field(default_factory=list)
    compatible_moods: List[str] = field(default_factory=list)
    best_transition_points: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON storage"""
        d = asdict(self)
        d['filepath'] = str(d['filepath'])
        d['color_histogram'] = d['color_histogram'].tolist() if d['color_histogram'] is not None else None
        d['frame_embeddings'] = d['frame_embeddings'].tolist() if d['frame_embeddings'] is not None else None
        d['motion_vector'] = d['motion_vector'].tolist() if d['motion_vector'] is not None else None
        return d


class ClipAnalyzer:
    """Comprehensive clip analysis engine"""
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
    
    def analyze_clip(self, clip_path: Path) -> Optional[ClipMetadata]:
        """Perform complete clip analysis"""
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(clip_path))
            if not cap.isOpened():
                return None
            
            # Extract basic info
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Get codec
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            # File info
            file_size = clip_path.stat().st_size / (1024 * 1024)
            file_hash = self._compute_file_hash(clip_path)
            
            # Aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            aspect_label = self._get_aspect_ratio_label(aspect_ratio)
            
            # Quality estimation
            quality_label, quality_score = self._estimate_quality(width, height, fps)
            
            # Analyze clip
            motion = self._analyze_motion(cap, fps, total_frames)
            color_analysis = self._analyze_colors(cap, total_frames)
            content = self._analyze_content(cap, total_frames)
            audio = self._analyze_audio(clip_path)
            scenes = self._detect_scenes(cap, total_frames)
            
            # Extract keyframes and embeddings
            keyframes = self._extract_keyframes(cap, total_frames, num_frames=5)
            color_hist = self._compute_color_histogram(keyframes)
            frame_emb = self._compute_frame_embeddings(keyframes)
            
            cap.release()
            
            # Create metadata
            clip_id = hashlib.md5(str(clip_path).encode()).hexdigest()[:12]
            
            metadata = ClipMetadata(
                clip_id=clip_id,
                filename=clip_path.name,
                filepath=clip_path,
                file_hash=file_hash,
                file_size_mb=file_size,
                added_date=datetime.now().isoformat(),
                duration_seconds=duration,
                width=width,
                height=height,
                fps=int(fps),
                total_frames=total_frames,
                codec=codec,
                bitrate_kbps=None,
                aspect_ratio=aspect_ratio,
                aspect_ratio_label=aspect_label,
                estimated_quality=quality_label,
                quality_score=quality_score,
                motion=motion,
                color_analysis=color_analysis,
                content=content,
                audio=audio,
                scenes=scenes,
                color_histogram=color_hist,
                frame_embeddings=frame_emb
            )
            
            return metadata
        
        except Exception as e:
            print(f"Error analyzing {clip_path}: {e}")
            return None
    
    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_aspect_ratio_label(self, ratio: float) -> str:
        """Get human-readable aspect ratio label"""
        if abs(ratio - 16/9) < 0.05:
            return "16:9"
        elif abs(ratio - 4/3) < 0.05:
            return "4:3"
        elif abs(ratio - 1) < 0.05:
            return "1:1"
        elif abs(ratio - 21/9) < 0.05:
            return "21:9"
        elif abs(ratio - 9/16) < 0.05:
            return "9:16"
        else:
            return f"{ratio:.2f}:1"
    
    def _estimate_quality(self, width: int, height: int, fps: float) -> Tuple[str, float]:
        """Estimate video quality tier"""
        pixels = width * height
        
        if pixels >= 3840 * 2160:  # 4K
            return "4K", min(1.0, (pixels / (3840 * 2160)) * (fps / 60))
        elif pixels >= 1920 * 1080:  # 1080p
            return "1080p", min(1.0, (pixels / (1920 * 1080)) * (fps / 60))
        elif pixels >= 1280 * 720:  # 720p
            return "720p", min(1.0, (pixels / (1280 * 720)) * (fps / 60))
        else:
            return "SD", (pixels / (1280 * 720))
    
    def _analyze_motion(self, cap: cv2.VideoCapture, fps: float, total_frames: int) -> ClipMotion:
        """Analyze motion characteristics"""
        
        motion_scores = []
        prev_gray = None
        flow_vectors = []
        
        sample_size = min(30, total_frames)
        frame_indices = np.linspace(0, total_frames - 2, sample_size, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            small = cv2.resize(frame, (320, 180))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_score = np.mean(mag) / 10.0
                motion_scores.append(min(1.0, motion_score))
                flow_vectors.append(flow)
            
            prev_gray = gray
        
        avg_motion = np.mean(motion_scores) if motion_scores else 0.5
        
        # Detect camera type
        camera_type = self._detect_camera_type(flow_vectors)
        motion_direction = self._detect_motion_direction(flow_vectors)
        is_slow_motion = fps < 24
        is_timelapse = fps < 1
        
        return ClipMotion(
            motion_score=float(avg_motion),
            camera_type=camera_type,
            motion_direction=motion_direction,
            fps=int(fps),
            is_slow_motion=is_slow_motion,
            is_timelapse=is_timelapse
        )
    
    def _detect_camera_type(self, flow_vectors: List[np.ndarray]) -> str:
        """Detect camera movement type"""
        if not flow_vectors:
            return "static"
        
        # Analyze flow patterns
        horizontal_motion = []
        vertical_motion = []
        
        for flow in flow_vectors:
            h_motion = np.mean(np.abs(flow[..., 0]))
            v_motion = np.mean(np.abs(flow[..., 1]))
            horizontal_motion.append(h_motion)
            vertical_motion.append(v_motion)
        
        h_avg = np.mean(horizontal_motion)
        v_avg = np.mean(vertical_motion)
        
        if h_avg < 0.1 and v_avg < 0.1:
            return "static"
        elif h_avg > v_avg:
            return "pan"
        elif v_avg > h_avg:
            return "zoom"
        else:
            return "handheld"
    
    def _detect_motion_direction(self, flow_vectors: List[np.ndarray]) -> str:
        """Detect primary motion direction"""
        if not flow_vectors:
            return "none"
        
        # Compute dominant direction
        angles = []
        for flow in flow_vectors:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            angles.extend(ang[mag > 1].flatten())
        
        if not angles:
            return "none"
        
        # Compute circular mean
        angles = np.array(angles)
        avg_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
        avg_angle_deg = np.degrees(avg_angle) % 360
        
        if 45 <= avg_angle_deg <= 135:
            return "vertical"
        elif 135 <= avg_angle_deg <= 225:
            return "horizontal"
        elif 225 <= avg_angle_deg <= 315:
            return "vertical"
        else:
            return "horizontal"
    
    def _analyze_colors(self, cap: cv2.VideoCapture, total_frames: int) -> ClipColorAnalysis:
        """Comprehensive color analysis"""
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        saturations = []
        brightnesses = []
        contrasts = []
        all_colors = []
        
        sample_size = min(20, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, sample_size, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Resize for processing
            small = cv2.resize(frame, (100, 100))
            
            # HSV analysis
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:, :, 1]) / 255.0
            saturation = float(saturation)
            saturations.append(saturation)
            
            # Brightness
            v_channel = hsv[:, :, 2]
            brightness = np.mean(v_channel) / 255.0
            brightnesses.append(brightness)
            
            # Contrast
            contrast = np.std(v_channel) / 255.0
            contrasts.append(contrast)
            
            # Color palette
            pixels = small.reshape(-1, 3)
            all_colors.extend(pixels)
        
        # Get dominant colors
        all_colors = np.array(all_colors)
        if len(all_colors) > 0:
            # Simple clustering
            unique_colors = np.unique(all_colors.reshape(-1, 3), axis=0)[:5]
            dominant_colors = [tuple(map(int, c)) for c in unique_colors]
        else:
            dominant_colors = [(0, 0, 0)]
        
        # Color temperature
        avg_r = np.mean([c[2] for c in dominant_colors])
        avg_b = np.mean([c[0] for c in dominant_colors])
        if avg_r > avg_b + 30:
            color_temp = "warm"
        elif avg_b > avg_r + 30:
            color_temp = "cool"
        else:
            color_temp = "neutral"
        
        return ClipColorAnalysis(
            dominant_colors=dominant_colors,
            color_palette_count=len(set(dominant_colors)),
            saturation_avg=float(np.mean(saturations)),
            brightness_avg=float(np.mean(brightnesses)),
            contrast_avg=float(np.mean(contrasts)),
            color_temperature=color_temp,
            vibrancy_score=float(np.mean(saturations)),
            color_harmony="analogous"
        )
    
    def _analyze_content(self, cap: cv2.VideoCapture, total_frames: int) -> ClipContent:
        """Analyze clip content and semantics"""
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Simple heuristics (would be replaced with AI model in production)
        ret, frame = cap.read()
        
        if ret:
            height, width = frame.shape[:2]
            # Estimate quality
            if height >= 1080:
                quality = 1.0
            else:
                quality = 0.5
        else:
            quality = 0.5
        
        return ClipContent(
            subject_tags=["generic"],
            environment_type="unknown",
            lighting_quality="mixed",
            composition_style="mixed",
            visual_style="realistic",
            mood_tags=["neutral"],
            detected_objects={},
            scene_complexity=0.5,
            estimated_duration_use="medium_clip"
        )
    
    def _analyze_audio(self, clip_path: Path) -> ClipAudio:
        """Analyze audio track if present"""
        
        try:
            import librosa
            
            cap = cv2.VideoCapture(str(clip_path))
            audio_exists = cap.get(cv2.CAP_PROP_AUDIO_INDEX) >= 0
            cap.release()
            
            return ClipAudio(has_audio=audio_exists)
        except:
            return ClipAudio(has_audio=False)
    
    def _detect_scenes(self, cap: cv2.VideoCapture, total_frames: int) -> List[ClipScene]:
        """Detect scene boundaries"""
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        scenes = []
        prev_gray = None
        threshold = 30
        
        for frame_idx in range(0, total_frames, int(total_frames / 10)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = np.mean(diff)
                
                if mean_diff > threshold:
                    scenes.append(ClipScene(
                        start_frame=frame_idx - int(total_frames / 10),
                        end_frame=frame_idx,
                        label="transition",
                        confidence=min(1.0, mean_diff / 100),
                        color_palette=[],
                        brightness_avg=0.5,
                        contrast_avg=0.5
                    ))
            
            prev_gray = gray
        
        return scenes
    
    def _extract_keyframes(self, cap: cv2.VideoCapture, total_frames: int, num_frames: int = 5) -> List[np.ndarray]:
        """Extract evenly spaced keyframes"""
        
        keyframes = []
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                keyframes.append(frame)
        
        return keyframes
    
    def _compute_color_histogram(self, frames: List[np.ndarray]) -> np.ndarray:
        """Compute averaged color histogram"""
        
        histograms = []
        for frame in frames:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms.append(hist)
        
        if histograms:
            return np.mean(np.array(histograms), axis=0).astype(np.float32)
        else:
            return np.zeros(64, dtype=np.float32)
    
    def _compute_frame_embeddings(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """Compute semantic embeddings for frames"""
        
        if self.embedding_model is None:
            return None
        
        try:
            embeddings = self.embedding_model.encode_frames_batch(frames)
            return np.mean(embeddings, axis=0).astype(np.float32)
        except:
            return None


class ClipDatabase:
    """SQLite database for clip metadata"""
    
    def __init__(self, db_path: Path = Path("clips.db")):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS clips (
                clip_id TEXT PRIMARY KEY,
                filename TEXT,
                filepath TEXT,
                file_hash TEXT UNIQUE,
                file_size_mb REAL,
                added_date TEXT,
                duration_seconds REAL,
                width INTEGER,
                height INTEGER,
                fps INTEGER,
                total_frames INTEGER,
                codec TEXT,
                aspect_ratio REAL,
                aspect_ratio_label TEXT,
                estimated_quality TEXT,
                quality_score REAL,
                motion_score REAL,
                camera_type TEXT,
                scene_complexity REAL,
                times_used INTEGER,
                last_used_date TEXT,
                user_rating REAL,
                metadata_json TEXT
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS clip_tags (
                clip_id TEXT,
                tag TEXT,
                FOREIGN KEY(clip_id) REFERENCES clips(clip_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_clip(self, metadata: ClipMetadata) -> bool:
        """Add clip metadata to database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            metadata_json = json.dumps(metadata.to_dict(), default=str)
            
            c.execute('''
                INSERT OR REPLACE INTO clips VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.clip_id,
                metadata.filename,
                str(metadata.filepath),
                metadata.file_hash,
                metadata.file_size_mb,
                metadata.added_date,
                metadata.duration_seconds,
                metadata.width,
                metadata.height,
                metadata.fps,
                metadata.total_frames,
                metadata.codec,
                metadata.aspect_ratio,
                metadata.aspect_ratio_label,
                metadata.estimated_quality,
                metadata.quality_score,
                metadata.motion.motion_score,
                metadata.motion.camera_type,
                metadata.content.scene_complexity,
                metadata.times_used,
                metadata.last_used_date,
                metadata.user_rating,
                metadata_json
            ))
            
            # Add tags
            for tag in metadata.semantic_tags:
                c.execute('INSERT INTO clip_tags VALUES (?, ?)', (metadata.clip_id, tag))
            
            conn.commit()
            conn.close()
            return True
        
        except Exception as e:
            print(f"Database error: {e}")
            return False
    
    def get_clip(self, clip_id: str) -> Optional[Dict]:
        """Retrieve clip metadata by ID"""
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT metadata_json FROM clips WHERE clip_id = ?', (clip_id,))
        row = c.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        return None
    
    def search_clips(self, query: Dict) -> List[Dict]:
        """Search clips by criteria"""
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        sql = 'SELECT metadata_json FROM clips WHERE 1=1'
        params = []
        
        if 'min_duration' in query:
            sql += ' AND duration_seconds >= ?'
            params.append(query['min_duration'])
        
        if 'max_duration' in query:
            sql += ' AND duration_seconds <= ?'
            params.append(query['max_duration'])
        
        if 'quality' in query:
            sql += ' AND estimated_quality = ?'
            params.append(query['quality'])
        
        if 'aspect_ratio' in query:
            sql += ' AND aspect_ratio_label = ?'
            params.append(query['aspect_ratio'])
        
        if 'min_motion' in query:
            sql += ' AND motion_score >= ?'
            params.append(query['min_motion'])
        
        c.execute(sql, params)
        rows = c.fetchall()
        conn.close()
        
        return [json.loads(row[0]) for row in rows]
    
    def update_usage(self, clip_id: str) -> None:
        """Update clip usage statistics"""
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            UPDATE clips SET times_used = times_used + 1, last_used_date = ? WHERE clip_id = ?
        ''', (datetime.now().isoformat(), clip_id))
        
        conn.commit()
        conn.close()


class ClipInformationSystem:
    """Complete standalone clip information and management system"""
    
    def __init__(self, clips_dir: Path, db_path: Path = Path("clips.db"), embedding_model=None):
        self.clips_dir = Path(clips_dir)
        self.analyzer = ClipAnalyzer(embedding_model)
        self.database = ClipDatabase(db_path)
    
    def scan_and_analyze(self, force_rescan: bool = False) -> Dict[str, ClipMetadata]:
        """Scan directory and analyze all clips"""
        
        results = {}
        video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv'}
        
        clip_files = [f for f in self.clips_dir.iterdir() 
                     if f.is_file() and f.suffix.lower() in video_extensions]
        
        print(f"\n[Clip Info System] Analyzing {len(clip_files)} clips...")
        
        for clip_path in tqdm(clip_files, desc="Analyzing clips"):
            metadata = self.analyzer.analyze_clip(clip_path)
            
            if metadata:
                # Save to database
                self.database.add_clip(metadata)
                results[metadata.clip_id] = metadata
                print(f"  ✓ {clip_path.name}: {metadata.width}x{metadata.height}, {metadata.duration_seconds:.1f}s")
            else:
                print(f"  ✗ Failed: {clip_path.name}")
        
        return results
    
    def get_clip_info(self, clip_id: str) -> Optional[ClipMetadata]:
        """Get detailed information about a clip"""
        return self.database.get_clip(clip_id)
    
    def search(self, **kwargs) -> List[ClipMetadata]:
        """Search clips by criteria"""
        results = self.database.search_clips(kwargs)
        return results
    
    def get_compatible_clips(self, genre: str, mood: str, duration_range: Tuple[float, float]) -> List[ClipMetadata]:
        """Find clips compatible with specific music context"""
        
        results = self.database.search_clips({
            'min_duration': duration_range[0],
            'max_duration': duration_range[1]
        })
        
        return results
    
    def print_clip_report(self, metadata: ClipMetadata) -> None:
        """Print detailed clip analysis report"""
        
        print(f"\n{'='*70}")
        print(f"CLIP INFORMATION REPORT")
        print(f"{'='*70}")
        print(f"\n📁 FILE INFO:")
        print(f"  Filename: {metadata.filename}")
        print(f"  ID: {metadata.clip_id}")
        print(f"  Size: {metadata.file_size_mb:.1f} MB")
        print(f"  Added: {metadata.added_date}")
        
        print(f"\n📊 TECHNICAL SPECS:")
        print(f"  Resolution: {metadata.width}x{metadata.height} ({metadata.estimated_quality})")
        print(f"  Aspect Ratio: {metadata.aspect_ratio_label}")
        print(f"  Duration: {metadata.duration_seconds:.2f}s ({metadata.total_frames} frames)")
        print(f"  Frame Rate: {metadata.fps} fps")
        print(f"  Codec: {metadata.codec}")
        print(f"  Quality Score: {metadata.quality_score:.1%}")
        
        print(f"\n🎬 MOTION ANALYSIS:")
        print(f"  Motion Score: {metadata.motion.motion_score:.1%}")
        print(f"  Camera Type: {metadata.motion.camera_type}")
        print(f"  Motion Direction: {metadata.motion.motion_direction}")
        print(f"  Slow Motion: {metadata.motion.is_slow_motion}")
        
        print(f"\n🎨 COLOR ANALYSIS:")
        print(f"  Saturation: {metadata.color_analysis.saturation_avg:.1%}")
        print(f"  Brightness: {metadata.color_analysis.brightness_avg:.1%}")
        print(f"  Contrast: {metadata.color_analysis.contrast_avg:.1%}")
        print(f"  Color Temperature: {metadata.color_analysis.color_temperature}")
        print(f"  Vibrancy: {metadata.color_analysis.vibrancy_score:.1%}")
        print(f"  Dominant Colors: {metadata.color_analysis.dominant_colors}")
        
        print(f"\n📝 CONTENT ANALYSIS:")
        print(f"  Subject Tags: {', '.join(metadata.content.subject_tags)}")
        print(f"  Environment: {metadata.content.environment_type}")
        print(f"  Lighting: {metadata.content.lighting_quality}")
        print(f"  Visual Style: {metadata.content.visual_style}")
        print(f"  Scene Complexity: {metadata.content.scene_complexity:.1%}")
        print(f"  Mood Tags: {', '.join(metadata.content.mood_tags)}")
        
        print(f"\n🎵 MUSIC COMPATIBILITY:")
        print(f"  Compatible Genres: {', '.join(metadata.compatible_music_genres) if metadata.compatible_music_genres else 'None specified'}")
        print(f"  Compatible Moods: {', '.join(metadata.compatible_moods) if metadata.compatible_moods else 'None specified'}")
        
        print(f"\n📈 USAGE STATS:")
        print(f"  Times Used: {metadata.times_used}")
        print(f"  Last Used: {metadata.last_used_date or 'Never'}")
        print(f"  User Rating: {metadata.user_rating:.1f}/5.0")
        
        print(f"\n{'='*70}\n")
