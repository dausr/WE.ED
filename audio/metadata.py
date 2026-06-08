"""MP3 Metadata Extraction - Artist, Genre, Tags, etc."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from mutagen.mp3 import MP3
from mutagen.id3 import ID3


@dataclass
class MP3Metadata:
    """Extracted MP3 metadata"""
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None
    duration: Optional[float] = None
    bitrate: Optional[int] = None
    tags: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class MetadataExtractor:
    """Extract metadata from MP3 files using ID3 tags"""

    @staticmethod
    def extract(audio_path: str) -> MP3Metadata:
        """Extract all metadata from MP3 file"""
        
        try:
            # Load MP3 with ID3 tags
            audio_file = MP3(audio_path)
            
            try:
                tags = ID3(audio_path)
            except:
                tags = {}
            
            # Duration
            duration = audio_file.info.length
            bitrate = audio_file.info.bitrate
            
            # Extract tags
            title = MetadataExtractor._get_tag(tags, ['TIT2'], 'Unknown')
            artist = MetadataExtractor._get_tag(tags, ['TPE1'], 'Unknown')
            album = MetadataExtractor._get_tag(tags, ['TALB'], 'Unknown')
            genre = MetadataExtractor._get_tag(tags, ['TCON'], 'Unknown')
            year = MetadataExtractor._get_tag(tags, ['TDRC'], None)
            
            # Try to parse year as int
            year_int = None
            if year:
                try:
                    year_int = int(str(year)[:4])
                except:
                    pass
            
            return MP3Metadata(
                title=title,
                artist=artist,
                album=album,
                genre=genre,
                year=year_int,
                duration=duration,
                bitrate=bitrate,
                tags={str(k): str(v) for k, v in tags.items()}
            )
        
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return MP3Metadata()
    
    @staticmethod
    def _get_tag(tags: Dict, tag_keys: list, default: Any = None) -> Any:
        """Safely extract tag from ID3 dictionary"""
        for key in tag_keys:
            if key in tags:
                try:
                    return str(tags[key].text[0]) if hasattr(tags[key], 'text') else str(tags[key])
                except:
                    pass
        return default
