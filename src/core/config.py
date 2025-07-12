"""
Core configuration module for sports analytics system.

This module contains the main configuration class and system metrics
for the sports analytics application.
"""

import os
import json
import hashlib
import logging
import yaml
import psutil
import cv2
from typing import Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SystemMetrics:
    """Track system resource usage"""
    cpu_percent: float
    memory_percent: float
    available_memory_gb: float


class Config:
    """Configuration with parameter hashing for robust checkpointing."""
    # Configuration values (will be overridden by YAML)
    VIDEO_PATH = 'sample_video.mp4'
    OUTPUT_CSV_PATH = 'enterprise_analytics_output.csv'
    CHECKPOINT_DIR = './analyzer_checkpoints'
    MODEL_PARAMS = {
        'TEAM_N_CLUSTERS': 3,
        'TEAM_FEATURE_BINS': 16,
        'MIN_CROP_SIZE': 100,
        'MAX_CROPS_PER_CLUSTER': 15,
        'FRAME_SKIP_INTERVAL': 2,
        'MAX_PROCESSING_QUEUE_SIZE': 32,
    }
    YOLO_MODEL_PATH = 'yolov8n.pt'
    SEGMENTATION_MODEL_PATH = 'path/to/your/segmentation_model.pth'
    JERSEY_YOLO_MODEL_PATH = None  # Path to custom YOLO model trained for jersey number detection
    TEAM_ROSTER_PATH = None  # Path to CSV file with team rosters (team_name, jersey_number, player_name)
    TEAM_SAMPLES_TO_COLLECT = 300
    HOMOGRAPHY_RECAL_THRESHOLD = 3.0
    HOMOGRAPHY_CHECK_INTERVAL = 150
    ACTION_SEQUENCE_LENGTH = 30
    FRAME_QUEUE_SIZE = 64
    PROCESSING_TIMEOUT = 1.0
    MAX_RETRIES = 3
    MIN_VIDEO_RESOLUTION = (320, 240)
    MAX_VIDEO_RESOLUTION = (4096, 2160)
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
    ENABLE_HOMOGRAPHY = True  # Set to False to disable homography calibration and use fallback transformation

    # Path generated dynamically based on video and parameter hashes
    checkpoint_path_prefix: Optional[str] = None
    
    @classmethod
    def load_from_yaml(cls, yaml_path: str = 'config/config.yaml') -> None:
        """Load configuration from YAML file."""
        try:
            if not os.path.exists(yaml_path):
                logging.warning(f"Config file {yaml_path} not found. Using default values.")
                return
                
            with open(yaml_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            # Update video settings
            if 'video' in config_data:
                cls.VIDEO_PATH = config_data['video'].get('path', cls.VIDEO_PATH)
                cls.SUPPORTED_VIDEO_FORMATS = config_data['video'].get('supported_formats', cls.SUPPORTED_VIDEO_FORMATS)
                cls.MIN_VIDEO_RESOLUTION = tuple(config_data['video'].get('min_resolution', cls.MIN_VIDEO_RESOLUTION))
                cls.MAX_VIDEO_RESOLUTION = tuple(config_data['video'].get('max_resolution', cls.MAX_VIDEO_RESOLUTION))
            
            # Update output settings
            if 'output' in config_data:
                cls.OUTPUT_CSV_PATH = config_data['output'].get('csv_path', cls.OUTPUT_CSV_PATH)
                cls.CHECKPOINT_DIR = config_data['output'].get('checkpoint_dir', cls.CHECKPOINT_DIR)
            
            # Update model paths
            if 'models' in config_data:
                cls.YOLO_MODEL_PATH = config_data['models'].get('yolo_path', cls.YOLO_MODEL_PATH)
                cls.SEGMENTATION_MODEL_PATH = config_data['models'].get('segmentation_path', cls.SEGMENTATION_MODEL_PATH)
                cls.JERSEY_YOLO_MODEL_PATH = config_data['models'].get('jersey_yolo_path', cls.JERSEY_YOLO_MODEL_PATH)
                cls.TEAM_ROSTER_PATH = config_data['models'].get('team_roster_path', cls.TEAM_ROSTER_PATH)
            
            # Update processing parameters
            if 'processing' in config_data:
                proc = config_data['processing']
                cls.MODEL_PARAMS['TEAM_N_CLUSTERS'] = proc.get('team_n_clusters', cls.MODEL_PARAMS['TEAM_N_CLUSTERS'])
                cls.MODEL_PARAMS['TEAM_FEATURE_BINS'] = proc.get('team_feature_bins', cls.MODEL_PARAMS['TEAM_FEATURE_BINS'])
                cls.MODEL_PARAMS['MIN_CROP_SIZE'] = proc.get('min_crop_size', cls.MODEL_PARAMS['MIN_CROP_SIZE'])
                cls.MODEL_PARAMS['MAX_CROPS_PER_CLUSTER'] = proc.get('max_crops_per_cluster', cls.MODEL_PARAMS['MAX_CROPS_PER_CLUSTER'])
                cls.MODEL_PARAMS['FRAME_SKIP_INTERVAL'] = proc.get('frame_skip_interval', cls.MODEL_PARAMS['FRAME_SKIP_INTERVAL'])
                cls.MODEL_PARAMS['MAX_PROCESSING_QUEUE_SIZE'] = proc.get('max_processing_queue_size', cls.MODEL_PARAMS['MAX_PROCESSING_QUEUE_SIZE'])
                cls.TEAM_SAMPLES_TO_COLLECT = proc.get('team_samples_to_collect', cls.TEAM_SAMPLES_TO_COLLECT)
                cls.HOMOGRAPHY_RECAL_THRESHOLD = proc.get('homography_recal_threshold', cls.HOMOGRAPHY_RECAL_THRESHOLD)
                cls.HOMOGRAPHY_CHECK_INTERVAL = proc.get('homography_check_interval', cls.HOMOGRAPHY_CHECK_INTERVAL)
                cls.ACTION_SEQUENCE_LENGTH = proc.get('action_sequence_length', cls.ACTION_SEQUENCE_LENGTH)
                cls.PROCESSING_TIMEOUT = proc.get('processing_timeout', cls.PROCESSING_TIMEOUT)
                cls.MAX_RETRIES = proc.get('max_retries', cls.MAX_RETRIES)
                cls.ENABLE_HOMOGRAPHY = proc.get('enable_homography', cls.ENABLE_HOMOGRAPHY)

                # Performance optimization flags
                cls.ENABLE_JERSEY_DETECTION = proc.get('enable_jersey_detection', True)
                cls.ENABLE_OCR = proc.get('enable_ocr', True)

                
            logging.info("Configuration loaded from YAML file")
            
        except Exception as e:
            logging.error(f"Error loading config from YAML: {e}")

    @classmethod
    def generate_checkpoint_prefix(cls, video_hash: str) -> None:
        """Creates a prefix based on video and parameter hashes for robust state management."""
        # Create a stable string representation of the parameters dictionary
        params_str = json.dumps(cls.MODEL_PARAMS, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        cls.checkpoint_path_prefix = os.path.join(cls.CHECKPOINT_DIR, f'{video_hash}_{params_hash}')
        logging.info(f"Using checkpoint prefix for this configuration: {cls.checkpoint_path_prefix}")

    @classmethod
    def validate_config(cls) -> bool:
        """Enhanced configuration validation."""
        # Check video file existence and format (skip if null - will be set at runtime)
        if cls.VIDEO_PATH is not None:
            if not os.path.exists(cls.VIDEO_PATH):
                logging.error(f"Video file not found: {cls.VIDEO_PATH}")
                return False

            # Validate video file is readable
            try:
                test_cap = cv2.VideoCapture(cls.VIDEO_PATH)
                if not test_cap.isOpened():
                    logging.error(f"Cannot open video file: {cls.VIDEO_PATH}")
                    return False
                test_cap.release()
            except Exception as e:
                logging.error(f"Video validation failed: {e}")
                return False

            video_ext = Path(cls.VIDEO_PATH).suffix.lower()
            if video_ext not in cls.SUPPORTED_VIDEO_FORMATS:
                logging.warning(f"Unsupported video format: {video_ext}")
        else:
            logging.info("Video path is null - will be set at runtime via upload")

        # Check available memory
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory < 2.0:
            logging.warning(f"Low available memory: {available_memory:.1f}GB")

        # Validate model parameters
        if cls.MODEL_PARAMS['TEAM_N_CLUSTERS'] < 2:
            logging.error("TEAM_N_CLUSTERS must be at least 2")
            return False

        output_dir = os.path.dirname(cls.OUTPUT_CSV_PATH) or '.'
        if not os.access(output_dir, os.W_OK):
            logging.error(f"Output directory is not writable: {output_dir}")
            return False

        return True

    @classmethod
    def get_system_metrics(cls) -> SystemMetrics:
        """Get current system resource usage"""
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            available_memory_gb=psutil.virtual_memory().available / (1024**3)
        )
