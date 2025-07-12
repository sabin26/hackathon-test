#!/usr/bin/env python3
"""
Main entry point for the Sports Analytics System.

This script provides a unified entry point for running the sports analytics
dashboard with the new modular architecture.
"""

import sys
import logging
import argparse
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.logging_utils import setup_logging, setup_warnings, setup_environment
from src.api.dashboard_server import DashboardServer


def main():
    """Main entry point for the sports analytics dashboard."""
    # Set up environment and logging
    setup_environment()
    setup_warnings()
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Sports Analytics Dashboard")
    parser.add_argument("--host", default="localhost", help="Dashboard server host")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard server port")
    parser.add_argument("--config", default="config/config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    try:
        # Create and start dashboard server
        dashboard = DashboardServer(host=args.host, port=args.port)
        
        logging.info("=" * 60)
        logging.info("SPORTS ANALYTICS DASHBOARD STARTING")
        logging.info("=" * 60)
        logging.info(f"Dashboard URL: http://{args.host}:{args.port}")
        logging.info("Upload a video file to begin analysis")
        logging.info("Press Ctrl+C to stop")
        logging.info("=" * 60)
        
        dashboard.run()
        return 0
        
    except KeyboardInterrupt:
        logging.info("Dashboard server stopped by user")
        return 0
    except Exception as e:
        logging.error(f"Failed to start dashboard: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
