#!/usr/bin/env python3
"""
Video analysis script for the Sports Analytics System.

This script allows direct video analysis without the web dashboard.
"""

import sys
import logging
import argparse
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logging_utils import setup_logging, setup_warnings, setup_environment
from src.core.config import Config
from src.analytics.sports_analyzer import VideoProcessor


def main():
    """Main entry point for video analysis."""
    # Set up environment and logging
    setup_environment()
    setup_warnings()
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Sports Video Analysis")
    parser.add_argument("video_path", help="Path to the video file to analyze")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    parser.add_argument("--output", default="analysis_results.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = Config()
        config.load_from_yaml(args.config)
        config.VIDEO_PATH = args.video_path
        config.OUTPUT_CSV_PATH = args.output
        
        # Validate configuration
        if not config.validate_config():
            logging.error("Configuration validation failed")
            return 1
        
        # Create video processor
        processor = VideoProcessor(config, enable_streaming=False)
        
        logging.info("=" * 60)
        logging.info("SPORTS VIDEO ANALYSIS STARTING")
        logging.info("=" * 60)
        logging.info(f"Video file: {args.video_path}")
        logging.info(f"Output file: {args.output}")
        logging.info("Press Ctrl+C to stop")
        logging.info("=" * 60)
        
        # Start processing
        processor.start()
        
        # Get results
        results = processor.get_results()
        logging.info(f"Analysis completed. Processed {len(results)} frames.")
        
        return 0
        
    except KeyboardInterrupt:
        logging.info("Analysis stopped by user")
        return 0
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
