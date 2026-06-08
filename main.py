#!/usr/bin/env python3
"""WE.ED Framework - Main CLI Entry Point"""

import click
import sys
from pathlib import Path
from config.settings import WEEDConfig, set_config
from pipeline import MusicVideoGenerator


@click.command()
@click.option(
    '--song',
    type=click.Path(exists=True),
    required=True,
    help='Path to MP3 song file'
)
@click.option(
    '--clips-dir',
    type=click.Path(exists=True),
    required=True,
    help='Directory containing video clips'
)
@click.option(
    '--output',
    type=click.Path(),
    default='./output',
    help='Output directory for rendered video'
)
@click.option(
    '--config',
    type=click.Path(exists=True),
    help='Optional YAML config file'
)
@click.option(
    '--fps',
    type=int,
    default=30,
    help='Output frames per second'
)
@click.option(
    '--resolution',
    type=str,
    default='1920x1080',
    help='Output resolution (WxH)'
)
@click.option(
    '--verbose',
    is_flag=True,
    help='Verbose output'
)
def main(song, clips_dir, output, config, fps, resolution, verbose):
    """
    🎬 WE.ED - AI-Powered Beat-Sync Music Video Creator
    
    Creates professional music videos with:
    - Semantic clip selection from your clip pool
    - Perfect beat synchronization
    - Professional transitions and effects
    - Smart versioning
    """
    
    try:
        click.echo("\n🎬 WE.ED Music Video Generator\n")
        
        # Load or create configuration
        if config:
            cfg = WEEDConfig.from_yaml(config)
            click.echo(f"✓ Loaded config from {config}")
        else:
            cfg = WEEDConfig(
                sounds_dir=Path(song).parent,
                clips_dir=Path(clips_dir),
                output_dir=Path(output)
            )
            cfg.video.frame_rate = fps
            cfg.video.resolution = resolution
            click.echo("✓ Using default configuration")
        
        set_config(cfg)
        
        # Create output directory
        Path(output).mkdir(parents=True, exist_ok=True)
        
        # Initialize and run generator
        generator = MusicVideoGenerator(config=cfg, verbose=verbose)
        output_path = generator.generate(song)
        
        click.echo(f"\n✅ Success! Video created: {output_path}\n")
        return 0
    
    except Exception as e:
        click.echo(f"\n❌ Error: {e}\n", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
