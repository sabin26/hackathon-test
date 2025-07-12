"""
Sports analyzer module for comprehensive video analysis.

This module contains the main VideoProcessor class that orchestrates
the entire sports analytics pipeline.
"""

import os
import time
import logging
import threading
import queue
import signal
from typing import List, Dict, Any, Optional
from collections import deque

from src.core.config import Config
from src.models.segmentation_model import SegmentationModel
from src.models.object_tracker import ObjectTracker
from src.core.homography_manager import HomographyManager
from src.analytics.team_identifier import TeamIdentifier
from src.analytics.jersey_detector import JerseyNumberDetector
from src.utils.performance_utils import PerformanceMonitor
from src.utils.video_utils import validate_video_file, get_video_properties, create_video_hash


class VideoProcessor:
    """The main orchestrator for the entire analytics pipeline."""

    def __init__(self, config: Config, enable_streaming: bool = False):
        self.config = config
        self.cap = None
        self.video_hash = None
        self.frame_queue = queue.Queue(maxsize=config.MODEL_PARAMS['MAX_PROCESSING_QUEUE_SIZE'])
        self.results_data: List[Dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.processing_lock = threading.Lock()
        self.frame_skip_counter = 0

        # Reset log file for new video processing session
        self._reset_log_file()

        # Real-time streaming support
        self.enable_streaming = enable_streaming
        self.dashboard_server: Optional[Any] = None
        if enable_streaming:
            logging.info("Real-time streaming enabled - dashboard server will be set externally")

        # Initialize video capture
        self._initialize_video()

        # Initialize components
        self._initialize_components()

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Add graceful shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _reset_log_file(self):
        """Reset the log file for a new video processing session."""
        try:
            # Clear the sports_analytics.log file by opening it in write mode
            with open('sports_analytics.log', 'w'):
                pass  # Just opening in 'w' mode clears the file
            logging.info("Log file reset for new video processing session")
        except Exception as e:
            logging.warning(f"Failed to reset log file: {e}")

    def _signal_handler(self, signum, _):
        """Handle shutdown signals gracefully"""
        logging.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_event.set()

    def _initialize_video(self) -> None:
        """Initialize video capture with error handling."""
        try:
            # Skip video initialization if no path provided (will be set at runtime)
            if not self.config.VIDEO_PATH:
                logging.info("No video path provided - video will be set at runtime")
                self.cap = None
                self.fps = 30  # Default FPS
                return

            if not validate_video_file(self.config.VIDEO_PATH):
                raise ValueError(f"Invalid video file: {self.config.VIDEO_PATH}")

            # Get video properties
            properties = get_video_properties(self.config.VIDEO_PATH)
            self.fps = properties['fps']
            self.total_frames = properties['frame_count']
            
            # Create video hash for checkpointing
            self.video_hash = create_video_hash(self.config.VIDEO_PATH)
            self.config.generate_checkpoint_prefix(self.video_hash)

            logging.info(f"Video initialized: {properties['width']}x{properties['height']} @ {self.fps:.1f}fps")
            logging.info(f"Total frames: {self.total_frames}, Duration: {properties['duration']:.1f}s")

        except Exception as e:
            logging.error(f"Video initialization failed: {e}")
            raise

    def _initialize_components(self) -> None:
        """Initialize all processing components."""
        try:
            # Initialize segmentation model
            self.segmentation_model = SegmentationModel(self.config.SEGMENTATION_MODEL_PATH)
            
            # Initialize object tracker
            self.object_tracker = ObjectTracker(self.config.YOLO_MODEL_PATH)
            
            # Initialize homography manager
            pitch_dims = (105.0, 68.0)  # Standard football pitch dimensions in meters
            self.homography_manager = HomographyManager(pitch_dims, self.segmentation_model)
            
            # Initialize team identifier
            self.team_identifier = TeamIdentifier(
                self.config.MODEL_PARAMS['TEAM_N_CLUSTERS'],
                config=self.config
            )
            
            # Initialize jersey number detector
            self.jersey_detector = JerseyNumberDetector(
                self.config.JERSEY_YOLO_MODEL_PATH,
                config=self.config
            )
            
            logging.info("All components initialized successfully")
            
        except Exception as e:
            logging.error(f"Component initialization failed: {e}")
            raise

    def set_video_path(self, video_path: str) -> bool:
        """Set video path at runtime (for uploaded videos)."""
        try:
            if not validate_video_file(video_path):
                logging.error(f"Invalid video file: {video_path}")
                return False
                
            self.config.VIDEO_PATH = video_path
            self._initialize_video()
            return True
            
        except Exception as e:
            logging.error(f"Failed to set video path: {e}")
            return False

    def start(self) -> None:
        """Start the video processing pipeline."""
        try:
            if not self.config.VIDEO_PATH:
                logging.error("No video path set - cannot start processing")
                return
                
            logging.info("Starting video processing pipeline...")
            self._process_video()
            
        except Exception as e:
            logging.error(f"Video processing failed: {e}")
        finally:
            self.cleanup()

    def _process_video(self) -> None:
        """Main video processing loop."""
        import cv2
        
        try:
            self.cap = cv2.VideoCapture(self.config.VIDEO_PATH)
            if not self.cap.isOpened():
                raise ValueError("Could not open video file")
                
            frame_count = 0
            
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    logging.info("End of video reached")
                    break
                    
                start_time = time.time()
                
                # Process frame
                self._process_frame(frame, frame_count)
                
                # Record performance
                processing_time = time.time() - start_time
                self.performance_monitor.record_frame_processed(processing_time)
                
                frame_count += 1
                
                # Log progress periodically
                if frame_count % 100 == 0:
                    logging.info(f"Processed {frame_count} frames")
                    
        except Exception as e:
            logging.error(f"Video processing error: {e}")
            self.performance_monitor.record_error()
        finally:
            if self.cap:
                self.cap.release()

    def _process_frame(self, frame, frame_count: int) -> None:
        """Process a single frame."""
        try:
            # Skip frames based on configuration
            if frame_count % self.config.MODEL_PARAMS['FRAME_SKIP_INTERVAL'] != 0:
                self.performance_monitor.record_frame_skipped()
                return
                
            # Track objects
            tracked_objects = self.object_tracker.track_objects(frame)
            
            # Apply homography transformation
            tracked_objects = self.homography_manager.apply_homography(tracked_objects)
            
            # Collect team samples and detect jersey numbers
            for obj in tracked_objects:
                if obj['type'] == 'person':
                    # Collect team identification samples
                    self.team_identifier.collect_and_store_sample(frame, obj['bbox_video'])
                    
                    # Detect jersey numbers
                    jersey_number = self.jersey_detector.detect_jersey_number(
                        frame, obj['bbox_video'], obj['id']
                    )
                    obj['jersey_number'] = jersey_number
                    
                    # Classify team
                    team = self.team_identifier.classify_player(frame, obj['bbox_video'])
                    obj['team'] = team
            
            # Store results
            frame_data = {
                'frame_number': frame_count,
                'timestamp': time.time(),
                'objects': tracked_objects
            }
            
            self.results_data.append(frame_data)
            
            # Stream data if enabled
            if self.enable_streaming and self.dashboard_server:
                self._stream_frame_data(frame_data)
                
        except Exception as e:
            logging.error(f"Frame processing error: {e}")
            self.performance_monitor.record_error()

    def _stream_frame_data(self, frame_data: Dict[str, Any]) -> None:
        """Stream frame data to dashboard if enabled."""
        try:
            if hasattr(self.dashboard_server, 'data_queue'):
                self.dashboard_server.data_queue.put_nowait(frame_data)
        except Exception as e:
            logging.warning(f"Failed to stream frame data: {e}")

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.cap:
                self.cap.release()
                
            # Log performance summary
            self.performance_monitor.log_performance_summary()
            
            logging.info("Video processor cleanup completed")
            
        except Exception as e:
            logging.warning(f"Cleanup error: {e}")

    def get_results(self) -> List[Dict[str, Any]]:
        """Get processing results."""
        return self.results_data
