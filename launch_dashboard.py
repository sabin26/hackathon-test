#!/usr/bin/env python3
"""
Simple launcher for the Sports Analytics Dashboard

This script provides a simpler way to launch the dashboard with proper
async handling and better error management.
"""

import asyncio
import logging
import threading
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def run_video_processor(config_path="config.yaml"):
    """Run video processor in a separate thread"""
    try:
        from ai import Config, VideoProcessor
        
        # Load configuration
        config = Config()
        config.load_from_yaml(config_path)
        
        if not config.validate_config():
            logger.error("Configuration validation failed")
            return
        
        # Create processor without streaming (we'll handle that separately)
        processor = VideoProcessor(config, enable_streaming=False)
        
        # Start processing
        logger.info("Starting video processing...")
        processor.start()
        logger.info("Video processing completed")
        
    except Exception as e:
        logger.error(f"Video processing failed: {e}")

async def run_dashboard_with_data():
    """Run dashboard server and simulate data streaming"""
    try:
        from dashboard_server import DashboardServer
        import random
        
        # Create dashboard server
        server = DashboardServer(host="localhost", port=8000)
        
        # Start server in background
        _ = asyncio.create_task(server.start_server())
        
        # Wait a bit for server to start
        await asyncio.sleep(2)
        
        logger.info("Dashboard server started at http://localhost:8000")
        logger.info("Simulating real-time data...")
        
        # Simulate some data for demonstration
        frame_id = 0
        while True:
            # Create sample frame data
            sample_data = {
                "frame_id": frame_id,
                "timestamp": frame_id * 0.04,  # 25 FPS
                "objects": [
                    {
                        "id": 1,
                        "type": "person",
                        "team_name": "team_A",
                        "jersey_number": 10,
                        "pos_pitch": [random.uniform(10, 90), random.uniform(10, 50)],
                        "bbox_video": [100, 100, 150, 200]
                    },
                    {
                        "id": 2,
                        "type": "person", 
                        "team_name": "team_B",
                        "jersey_number": 7,
                        "pos_pitch": [random.uniform(10, 90), random.uniform(10, 50)],
                        "bbox_video": [200, 150, 250, 250]
                    },
                    {
                        "id": 3,
                        "type": "sports ball",
                        "pos_pitch": [random.uniform(20, 80), random.uniform(20, 40)],
                        "bbox_video": [300, 200, 320, 220]
                    }
                ],
                "actions": {
                    "1": {
                        "event_type": random.choice(["Possession", "Pass", "Shot_Attempt", ""]),
                        "event_outcome": "active",
                        "team_name": "team_A"
                    }
                }
            }
            
            # Broadcast to dashboard
            await server.broadcast_frame_data(sample_data)
            
            frame_id += 1
            await asyncio.sleep(0.1)  # 10 FPS for demo
            
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Dashboard error: {e}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Sports Analytics Dashboard")
    parser.add_argument("--mode", choices=["demo", "video"], default="demo",
                       help="Run in demo mode (simulated data) or video mode (process actual video)")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        logger.info("Starting in DEMO mode with simulated data")
        logger.info("Open http://localhost:8000 in your browser")
        logger.info("Press Ctrl+C to stop")
        
        try:
            asyncio.run(run_dashboard_with_data())
        except KeyboardInterrupt:
            logger.info("Demo stopped")
    
    elif args.mode == "video":
        logger.info("Starting in VIDEO mode")
        
        # Check if config file exists
        if not Path(args.config).exists():
            logger.error(f"Config file not found: {args.config}")
            return 1
        
        # Start video processing in a thread
        video_thread = threading.Thread(
            target=run_video_processor, 
            args=(args.config,),
            name="VideoProcessor"
        )
        video_thread.daemon = True
        video_thread.start()
        
        # Start dashboard
        try:
            asyncio.run(run_dashboard_with_data())
        except KeyboardInterrupt:
            logger.info("Stopped by user")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
