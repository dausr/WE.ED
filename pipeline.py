"""Main Orchestration Pipeline for WE.ED Framework"""

from pathlib import Path
from typing import Optional, List
from config.settings import WEEDConfig
from audio.analyzer import AudioAnalyzer
from audio.metadata import MetadataExtractor


class MusicVideoGenerator:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: WEEDConfig, verbose: bool = False):
        self.config = config
        self.verbose = verbose
        self.audio_analyzer = AudioAnalyzer(
            sample_rate=config.audio.sample_rate,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length
        )
        self.metadata_extractor = MetadataExtractor()
    
    def generate(self, song_path: str) -> str:
        """Generate music video from MP3 and clip pool"""
        
        song_path = Path(song_path)
        if not song_path.exists():
            raise FileNotFoundError(f"Song not found: {song_path}")
        
        if self.verbose:
            print(f"[Pipeline] Processing: {song_path}")
        
        # Step 1: Extract metadata
        if self.verbose:
            print("[Pipeline] Step 1: Extracting metadata...")
        metadata = self.metadata_extractor.extract(str(song_path))
        if self.verbose:
            print(f"  - Title: {metadata.title}")
            print(f"  - Artist: {metadata.artist}")
            print(f"  - Genre: {metadata.genre}")
        
        # Step 2: Audio analysis
        if self.verbose:
            print("[Pipeline] Step 2: Analyzing audio...")
        audio_analysis = self.audio_analyzer.analyze(str(song_path))
        if self.verbose:
            print(f"  - BPM: {audio_analysis.bpm:.1f}")
            print(f"  - Duration: {audio_analysis.duration:.1f}s")
            print(f"  - Beats detected: {len(audio_analysis.beats)}")
            print(f"  - Genre: {audio_analysis.genre_estimate.value}")
        
        # TODO: Step 3-6: Continue pipeline
        # 3. Load and preprocess clips
        # 4. Semantic analysis & matching
        # 5. Generate timeline with transitions
        # 6. Render video
        
        # Placeholder output
        output_name = self._generate_output_name(metadata.title)
        output_path = self.config.output_dir / output_name
        if self.verbose:
            print(f"[Pipeline] Output: {output_path}")
        
        return str(output_path)
    
    def _generate_output_name(self, title: Optional[str]) -> str:
        """Generate output filename with smart versioning"""
        
        if title:
            base_name = title.replace(' ', '_').lower()
        else:
            base_name = "musicvideo"
        
        # Check for existing versions
        version = 1
        if self.config.output.auto_versioning:
            existing = list(self.config.output_dir.glob(f"{base_name}_v*.mp4"))
            if existing:
                version = len(existing) + 1
        
        output_name = f"{base_name}_v{version:03d}.{self.config.output.output_format}"
        return output_name
