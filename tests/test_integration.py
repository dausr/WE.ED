"""Integration Tests for WE.ED Framework"""

import pytest
from pathlib import Path
import tempfile
import numpy as np
from unittest.mock import Mock, patch

# Import modules to test
from config.settings import WEEDConfig
from audio.analyzer import AudioAnalyzer
from clips.pool import ClipPool, ClipMetadata
from clips.preprocessor import ClipPreprocessor, ClipFrame
from clips.selector import SemanticClipSelector
from semantics.analyzer import SongContextAnalyzer
from semantics.embeddings import EmbeddingGenerator
from semantics.matching import SemanticMatcher
from video.transitions import TransitionEffects, TransitionType, TransitionConfig


class TestConfiguration:
    """Test configuration system"""
    
    def test_default_config_creation(self):
        """Test creating default WEEDConfig"""
        config = WEEDConfig()
        assert config.audio.sample_rate == 44100
        assert config.video.resolution == "1920x1080"
        assert config.output.auto_versioning == True
    
    def test_config_yaml_loading(self):
        """Test loading config from YAML"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
audio:
  sample_rate: 48000
video:
  frame_rate: 60
""")
            f.flush()
            config = WEEDConfig.from_yaml(f.name)
            assert config.audio.sample_rate == 48000
            assert config.video.frame_rate == 60


class TestAudioAnalysis:
    """Test audio processing"""
    
    def test_audio_analyzer_initialization(self):
        """Test AudioAnalyzer initialization"""
        analyzer = AudioAnalyzer(sample_rate=44100, n_fft=2048)
        assert analyzer.sample_rate == 44100
        assert analyzer.n_fft == 2048
    
    def test_song_context_analyzer(self):
        """Test SongContextAnalyzer"""
        analyzer = SongContextAnalyzer()
        
        metadata = {'title': 'Test', 'genre': 'electronic'}
        audio_analysis = {'bpm': 128}
        
        context = analyzer.analyze(metadata, audio_analysis)
        assert context.title == 'Test'
        assert context.bpm == 128
        assert context.genre == 'electronic'


class TestClipManagement:
    """Test clip pool and preprocessing"""
    
    def test_clip_metadata_creation(self):
        """Test ClipMetadata creation"""
        metadata = ClipMetadata(
            path=Path("test.mp4"),
            filename="test.mp4",
            duration=10.5,
            width=1920,
            height=1080,
            fps=30,
            total_frames=315,
            codec="h264",
            file_size_mb=50.0
        )
        
        assert metadata.filename == "test.mp4"
        assert metadata.duration == 10.5
        assert metadata.width == 1920
    
    def test_clip_frame_normalization(self):
        """Test ClipFrame normalization"""
        frame_data = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame = ClipFrame(frame=frame_data, frame_idx=0, timestamp=0.0)
        
        normalized = frame.normalize()
        assert normalized.dtype == np.float32
        assert np.max(normalized) <= 1.0
    
    def test_clip_pool_creation(self):
        """Test ClipPool initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = ClipPool(Path(tmpdir))
            assert pool.clips_dir == Path(tmpdir)
            assert len(pool.clips) == 0


class TestSemanticAnalysis:
    """Test semantic matching and embeddings"""
    
    def test_semantic_clip_selector(self):
        """Test SemanticClipSelector initialization"""
        selector = SemanticClipSelector(embedding_dim=512)
        assert selector.embedding_dim == 512
        assert selector.diversity_weight == 0.3
    
    def test_semantic_matcher(self):
        """Test SemanticMatcher"""
        matcher = SemanticMatcher(embedding_weight=0.7, feature_weight=0.3)
        
        # Create dummy embeddings
        segment_emb = np.random.randn(512).astype(np.float32)
        clip_embs = [np.random.randn(512).astype(np.float32) for _ in range(5)]
        
        # Test embedding similarity
        score = matcher._embedding_similarity(segment_emb, clip_embs[0])
        assert 0 <= score <= 1
    
    def test_semantic_matcher_feature_similarity(self):
        """Test feature-based matching"""
        matcher = SemanticMatcher()
        
        features1 = {'brightness': 0.5, 'saturation': 0.6}
        features2 = {'brightness': 0.51, 'saturation': 0.61}
        
        score = matcher._feature_similarity(features1, features2)
        assert 0 <= score <= 1
        assert score > 0.9  # Should be very similar


class TestVideoEffects:
    """Test video transition effects"""
    
    def test_transition_effects_initialization(self):
        """Test TransitionEffects initialization"""
        effects = TransitionEffects(fps=30)
        assert effects.fps == 30
    
    def test_cross_fade_transition(self):
        """Test cross-fade transition"""
        effects = TransitionEffects(fps=30)
        
        frame1 = np.full((1080, 1920, 3), 0, dtype=np.uint8)
        frame2 = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        
        result = effects.cross_fade(frame1, frame2, progress=0.5)
        
        assert result.shape == (1080, 1920, 3)
        assert result.dtype == np.uint8
        # At 50% progress, should be ~128 (halfway between 0 and 255)
        assert 100 < result[0, 0, 0] < 150
    
    def test_wobble_zoom_transition(self):
        """Test wobble zoom transition"""
        effects = TransitionEffects(fps=30)
        
        frame1 = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        
        result = effects.wobble_zoom(frame1, frame2, progress=0.5)
        
        assert result.shape == (1080, 1920, 3)
        assert result.dtype == np.uint8
    
    def test_wipe_transition(self):
        """Test wipe transition"""
        effects = TransitionEffects(fps=30)
        
        frame1 = np.full((1080, 1920, 3), 0, dtype=np.uint8)
        frame2 = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        
        result = effects.wipe(frame1, frame2, progress=0.5, direction="left_to_right")
        
        assert result.shape == (1080, 1920, 3)
        # Should have mix of black and white
        assert 0 in result and 255 in result
    
    def test_stutter_transition(self):
        """Test stutter effect"""
        effects = TransitionEffects(fps=30)
        frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        
        result = effects.stutter(frame, progress=0.3)
        
        assert result.shape == (1080, 1920, 3)
        assert result.dtype == np.uint8


class TestIntegration:
    """Integration tests for complete pipeline"""
    
    @patch('librosa.load')
    @patch('librosa.beat.beat_track')
    def test_full_pipeline_flow(self, mock_beat_track, mock_load):
        """Test complete pipeline flow with mocked audio"""
        
        # Mock audio loading
        mock_load.return_value = (np.zeros(44100), 44100)
        mock_beat_track.return_value = (120, np.array([0, 512, 1024]))
        
        # Would test full pipeline here
        # This is a placeholder for more complex integration tests
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
