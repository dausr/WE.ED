"""Complete Integrated Pipeline - Orchestrates all WE.ED Framework components"""

import sys
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm
import shutil

from config.settings import WEEDConfig, get_config, set_config
from audio.analyzer import AudioAnalyzer
from audio.metadata import MetadataExtractor
from clips.pool import ClipPool
from clips.preprocessor import ClipPreprocessor
from clips.selector import SemanticClipSelector
from semantics.analyzer import SongContextAnalyzer, SemanticEmbeddingModel, SemanticSegmenter
from semantics.embeddings import EmbeddingGenerator, ContextEmbedding
from semantics.matching import SemanticMatcher
from video.editor import VideoEditor, TimelineClip
from video.transitions import TransitionEffects, TransitionConfig, TransitionType
from video.renderer import VideoRenderer, RenderConfig


class MusicVideoGenerator:
    """Main pipeline orchestrator - generates complete music videos"""
    
    def __init__(self, config: Optional[WEEDConfig] = None, verbose: bool = True):
        """
        Initialize the music video generator.
        
        Args:
            config: WEEDConfig object (uses default if None)
            verbose: Enable progress output
        """
        self.config = config or get_config()
        self.verbose = verbose
        
        # Initialize all components
        self._init_components()
    
    def _init_components(self):
        """Initialize all pipeline components"""
        
        if self.verbose:
            print("[Pipeline] Initializing components...")
        
        # Audio analysis
        self.audio_analyzer = AudioAnalyzer(
            sample_rate=self.config.audio.sample_rate,
            n_fft=self.config.audio.n_fft,
            hop_length=self.config.audio.hop_length
        )
        self.metadata_extractor = MetadataExtractor()
        
        # Clip management
        self.clip_pool = ClipPool(self.config.clips_dir)
        self.clip_preprocessor = ClipPreprocessor(
            target_width=int(self.config.video.resolution.split('x')[0]),
            target_height=int(self.config.video.resolution.split('x')[1]),
            preserve_aspect=not self.config.output.no_black_bars
        )
        
        # Semantic analysis
        self.embedding_model = SemanticEmbeddingModel(
            model_name=self.config.semantics.embedding_model,
            device=self.config.semantics.device
        )
        self.context_analyzer = SongContextAnalyzer()
        self.embeddings_generator = EmbeddingGenerator(self.embedding_model)
        self.context_embedding = ContextEmbedding(self.embedding_model)
        self.segmenter = SemanticSegmenter(self.embedding_model)
        self.semantic_matcher = SemanticMatcher(
            embedding_weight=0.7,
            feature_weight=0.3
        )
        self.clip_selector = SemanticClipSelector(
            embedding_dim=self.embedding_model.embedding_dim,
            diversity_weight=self.config.clip_selection.diversity_factor
        )
        
        # Video processing
        self.video_editor = VideoEditor()
        self.transition_effects = TransitionEffects(fps=self.config.video.frame_rate)
        self.video_renderer = VideoRenderer()
        
        if self.verbose:
            print("[Pipeline] ✓ All components initialized")
    
    def generate(self, song_path: str) -> Optional[Path]:
        """
        Generate complete music video from MP3 and clip pool.
        
        Args:
            song_path: Path to input MP3 file
        
        Returns:
            Path to output video or None if failed
        """
        
        song_path = Path(song_path)
        if not song_path.exists():
            raise FileNotFoundError(f"Song not found: {song_path}")
        
        try:
            if self.verbose:
                print(f"\n🎬 WE.ED Music Video Generation Pipeline")
                print(f"{'='*60}")
            
            # Step 1: Extract metadata
            if self.verbose:
                print("\n[1/8] Extracting song metadata...")
            metadata = self._extract_metadata(song_path)
            
            # Step 2: Analyze audio
            if self.verbose:
                print("[2/8] Analyzing audio (BPM, beats, features)...")
            audio_analysis = self._analyze_audio(song_path)
            
            # Step 3: Analyze song context
            if self.verbose:
                print("[3/8] Extracting song context (mood, energy, danceability)...")
            song_context = self._extract_song_context(metadata, audio_analysis)
            
            # Step 4: Load and preprocess clip pool
            if self.verbose:
                print("[4/8] Loading and preprocessing clip pool...")
            clips_metadata, clip_embeddings = self._load_and_preprocess_clips()
            
            # Step 5: Segment song and generate embeddings
            if self.verbose:
                print("[5/8] Segmenting song and generating embeddings...")
            segments = self._segment_song(audio_analysis, song_context)
            
            # Step 6: Match clips to segments
            if self.verbose:
                print("[6/8] Matching clips to song segments (semantic matching)...")
            timeline_clips = self._match_and_create_timeline(
                segments, clip_embeddings, clips_metadata, audio_analysis
            )
            
            # Step 7: Add transitions and effects
            if self.verbose:
                print("[7/8] Building video timeline with transitions...")
            # (Timeline clips already include transition configs)
            
            # Step 8: Render final video
            if self.verbose:
                print("[8/8] Rendering final video with audio...")
            output_path = self._render_video(timeline_clips, song_path)
            
            if output_path:
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"✅ Success! Video created: {output_path}")
                    print(f"{'='*60}\n")
                return output_path
            else:
                if self.verbose:
                    print(f"\n❌ Failed to render video")
                return None
        
        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def _extract_metadata(self, song_path: Path) -> dict:
        """Extract MP3 metadata"""
        metadata = self.metadata_extractor.extract(str(song_path))
        if self.verbose:
            print(f"   Title: {metadata.title}")
            print(f"   Artist: {metadata.artist}")
            print(f"   Genre: {metadata.genre}")
            print(f"   Duration: {metadata.duration:.1f}s")
        return metadata.to_dict()
    
    def _analyze_audio(self, song_path: Path) -> dict:
        """Analyze audio features"""
        analysis = self.audio_analyzer.analyze(str(song_path))
        if self.verbose:
            print(f"   BPM: {analysis.bpm:.1f}")
            print(f"   Duration: {analysis.duration:.1f}s")
            print(f"   Beats detected: {len(analysis.beats)}")
            print(f"   Genre estimate: {analysis.genre_estimate.value}")
        
        return {
            'bpm': analysis.bpm,
            'beats': analysis.beats.tolist(),
            'duration': analysis.duration,
            'genre': analysis.genre_estimate.value,
            'audio_analysis': analysis
        }
    
    def _extract_song_context(self, metadata: dict, audio_analysis: dict) -> object:
        """Extract semantic song context"""
        context = self.context_analyzer.analyze(metadata, audio_analysis)
        if self.verbose:
            print(f"   Mood: {context.mood}")
            print(f"   Energy: {context.energy_level:.1%}")
            print(f"   Danceability: {context.danceability:.1%}")
        return context
    
    def _load_and_preprocess_clips(self) -> tuple:
        """Load clip pool and generate embeddings"""
        
        # Load clips from directory
        clips = self.clip_pool.load_clips(use_cache=True)
        if self.verbose:
            stats = self.clip_pool.stats()
            print(f"   Clips loaded: {stats['total_clips']}")
            print(f"   Total duration: {stats['total_duration']:.1f}s")
            print(f"   16:9 clips: {stats.get('16_9_clips', 0)}")
        
        # Generate embeddings for all clips
        if self.verbose:
            print("   Generating clip embeddings...")
        
        clip_embeddings = []
        for clip in tqdm(clips, disable=not self.verbose, desc="   Embeddings"):
            try:
                embedding = self.embeddings_generator.generate_clip_embedding(clip.path)
                clip_embeddings.append(embedding)
                self.clip_pool.set_embedding(clip.filename, embedding)
            except Exception as e:
                if self.verbose:
                    print(f"   Warning: Failed to embed {clip.filename}: {e}")
                clip_embeddings.append(None)
        
        return clips, clip_embeddings
    
    def _segment_song(self, audio_analysis: dict, song_context: object) -> List[dict]:
        """Segment song into semantic sections"""
        
        segments = self.segmenter.segment_by_beats(
            audio_analysis['audio_analysis'].beats,
            song_context
        )
        
        if self.verbose:
            print(f"   Song segmented into {len(segments)} parts")
        
        return segments
    
    def _match_and_create_timeline(self, segments: List[dict], 
                                   clip_embeddings: List, 
                                   clips_metadata: list,
                                   audio_analysis: dict) -> List[TimelineClip]:
        """Match clips to segments and create timeline"""
        
        # Extract segment embeddings
        segment_embeddings = [s['embedding'] for s in segments]
        
        # Match clips to segments
        matches = self.semantic_matcher.match_clips_to_segments(
            segment_embeddings,
            clip_embeddings,
            num_matches=2  # 2 options per segment for diversity
        )
        
        # Rerank with diversity
        matches = self.semantic_matcher.rerank_with_diversity(matches, clip_embeddings)
        
        if self.verbose:
            print(f"   Matched {len(matches)} segment groups")
        
        # Create timeline with clips and transitions
        timeline_clips = []
        current_time = 0.0
        beat_duration = 60.0 / audio_analysis['bpm']  # Duration of one beat
        
        for seg_idx, segment_matches in enumerate(matches):
            if not segment_matches:
                continue
            
            # Select best match
            best_match = segment_matches[0]
            clip_idx = best_match.clip_idx
            
            if clip_idx >= len(clips_metadata):
                continue
            
            clip_meta = clips_metadata[clip_idx]
            
            # Use portion of clip
            clip_duration = min(beat_duration * 4, clip_meta.duration)  # Up to 4 beats per clip
            
            # Add transition config
            transition = TransitionConfig(
                transition_type=TransitionType.WOBBLE_ZOOM,
                duration=self.config.transitions.fade_duration,
                intensity=self.config.transitions.wobble_intensity,
                frequency=self.config.transitions.wobble_frequency
            )
            
            timeline_clip = TimelineClip(
                clip_path=clip_meta.path,
                start_time=current_time,
                duration=clip_duration,
                source_offset=0,
                transitions={'config': transition}
            )
            
            timeline_clips.append(timeline_clip)
            current_time += clip_duration
        
        if self.verbose:
            print(f"   Timeline created: {len(timeline_clips)} clips, {current_time:.1f}s total")
        
        return timeline_clips
    
    def _render_video(self, timeline_clips: List[TimelineClip], 
                     song_path: Path) -> Optional[Path]:
        """Render final video with audio"""
        
        # Generate output filename with smart versioning
        output_name = self._generate_output_filename()
        output_path = self.config.output_dir / output_name
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Assemble video timeline
        if self.verbose:
            print(f"   Assembling timeline ({len(timeline_clips)} clips)...")
        
        video_success = self.video_editor.assemble_timeline(
            timeline_clips,
            output_path,
            fps=self.config.video.frame_rate,
            width=int(self.config.video.resolution.split('x')[0]),
            height=int(self.config.video.resolution.split('x')[1])
        )
        
        if not video_success:
            if self.verbose:
                print(f"   ❌ Failed to assemble video timeline")
            return None
        
        # Replace audio with MP3
        if self.verbose:
            print(f"   Adding audio track...")
        
        temp_video = output_path.with_suffix('.temp.mp4')
        shutil.move(output_path, temp_video)
        
        audio_success = self.video_editor.replace_audio(temp_video, song_path, output_path)
        temp_video.unlink(missing_ok=True)
        
        if not audio_success:
            if self.verbose:
                print(f"   ⚠️  Warning: Failed to add audio, using video-only")
            if not output_path.exists():
                shutil.move(temp_video, output_path)
        
        # Validate output
        if output_path.exists() and output_path.stat().st_size > 0:
            if self.verbose:
                print(f"   ✓ Video rendered: {output_path.name}")
                print(f"   Size: {output_path.stat().st_size / (1024*1024):.1f} MB")
            return output_path
        else:
            if self.verbose:
                print(f"   ❌ Output file not created")
            return None
    
    def _generate_output_filename(self) -> str:
        """Generate output filename with smart versioning"""
        
        base_name = "musicvideo"
        version = 1
        
        if self.config.output.auto_versioning:
            existing = list(self.config.output_dir.glob(f"{base_name}_v*.mp4"))
            if existing:
                version = len(existing) + 1
        
        return f"{base_name}_v{version:03d}.{self.config.output.output_format}"
