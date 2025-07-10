#!/usr/bin/env python3
"""
Real-time Sports Analytics Dashboard Runner

This script runs both the video processor and the web dashboard server
to provide real-time sports analytics visualization.
"""

import asyncio
import logging
import threading
import signal
import sys

# Import our modules
from ai import Config, VideoProcessor
from dashboard_server import DashboardServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DashboardRunner:
    """Manages both the video processor and dashboard server"""
    
    def __init__(self, config_path: str = 'config.yaml', host: str = "localhost", port: int = 8000):
        self.config_path = config_path
        self.host = host
        self.port = port

        # Optional override for video path
        self.video_override = None
        
        # Components
        self.config = None
        self.video_processor = None
        self.dashboard_server = None
        
        # Threading
        self.video_thread = None
        self.server_task = None
        self.loop = None
        self.shutdown_event = threading.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
    def initialize(self):
        """Initialize all components"""
        try:
            # Load configuration
            self.config = Config()
            self.config.load_from_yaml(self.config_path)

            # Override video path if specified
            if self.video_override:
                self.config.VIDEO_PATH = self.video_override
            
            if not self.config.validate_config():
                logger.error("Configuration validation failed")
                return False
            
            # Initialize dashboard server
            self.dashboard_server = DashboardServer(self.host, self.port)
            
            # Initialize video processor with streaming enabled
            self.video_processor = VideoProcessor(self.config, enable_streaming=True)
            
            # Set the dashboard server reference for streaming
            if hasattr(self.video_processor, 'dashboard_server'):
                self.video_processor.dashboard_server = self.dashboard_server
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def start_video_processing(self):
        """Start video processing in a separate thread"""
        def video_worker():
            try:
                logger.info("Starting video processing...")
                if self.video_processor is not None:
                    self.video_processor.start()
                    logger.info("Video processing completed")
                else:
                    logger.error("Video processor is not initialized.")
            except Exception as e:
                logger.error(f"Video processing failed: {e}")
            finally:
                # Signal shutdown when video processing is done
                self.shutdown_event.set()
        
        self.video_thread = threading.Thread(target=video_worker, name="VideoProcessor")
        self.video_thread.daemon = True
        self.video_thread.start()
    
    async def start_dashboard_server(self):
        """Start the dashboard server"""
        try:
            logger.info(f"Starting dashboard server on http://{self.host}:{self.port}")
            if self.dashboard_server is not None:
                await self.dashboard_server.start_server()
            else:
                logger.error("Dashboard server is not initialized.")
        except Exception as e:
            logger.error(f"Dashboard server failed: {e}")
    
    async def run_async(self):
        """Run the complete system asynchronously"""
        try:
            # Start video processing in background thread
            self.start_video_processing()
            
            # Start dashboard server
            server_task = asyncio.create_task(self.start_dashboard_server())
            
            # Wait for shutdown signal or video processing completion
            while not self.shutdown_event.is_set():
                await asyncio.sleep(1)
                
                # Check if video thread is still alive
                if self.video_thread and not self.video_thread.is_alive():
                    logger.info("Video processing completed, keeping dashboard running...")
                    # Don't shutdown immediately, keep dashboard running for review
            
            # Cancel server task
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
                
        except Exception as e:
            logger.error(f"Runtime error: {e}")
    
    def run(self):
        """Run the complete system"""
        if not self.initialize():
            logger.error("Failed to initialize, exiting")
            return 1
        
        try:
            # Create and run event loop
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            logger.info("=" * 60)
            logger.info("SPORTS ANALYTICS DASHBOARD STARTING")
            logger.info("=" * 60)
            logger.info(f"Dashboard URL: http://{self.host}:{self.port}")
            if self.config and hasattr(self.config, "VIDEO_PATH"):
                logger.info(f"Video file: {self.config.VIDEO_PATH}")
            else:
                logger.info("Video file: Not specified in config")
            logger.info("Press Ctrl+C to stop")
            logger.info("=" * 60)
            
            self.loop.run_until_complete(self.run_async())
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return 1
        finally:
            self.cleanup()
        
        return 0
    
    def shutdown(self):
        """Shutdown all components"""
        logger.info("Shutting down...")
        self.shutdown_event.set()
        
        # Stop video processor
        if self.video_processor:
            try:
                self.video_processor.stop_event.set()
            except Exception as e:
                logger.warning(f"Error stopping video processor: {e}")
        
        # Wait for video thread to finish
        if self.video_thread and self.video_thread.is_alive():
            logger.info("Waiting for video processing to complete...")
            self.video_thread.join(timeout=5)
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.video_processor:
                self.video_processor.cleanup()
            
            if self.loop and not self.loop.is_closed():
                self.loop.close()
                
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

def main():
    """Main entry point - simplified for video upload functionality"""
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Sports Analytics Dashboard")
    parser.add_argument("--host", default="localhost", help="Dashboard server host")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard server port")

    args = parser.parse_args()

    try:
        # Create and start dashboard server (no video validation needed)
        dashboard = DashboardServer(host=args.host, port=args.port)

        logger.info("Starting Sports Analytics Dashboard...")
        logger.info(f"Dashboard will be available at: http://{args.host}:{args.port}")
        logger.info("Upload a video file to begin analysis")

        dashboard.run()
        return 0

    except KeyboardInterrupt:
        logger.info("Dashboard server stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Failed to start dashboard: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
