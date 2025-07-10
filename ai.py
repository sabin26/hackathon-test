from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np
import threading
import queue
import time
import os
import joblib
import hashlib
import logging
import json
import torch
import yaml

# Set environment variable to disable pin_memory on MPS (Apple Silicon)
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from torchvision import transforms
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import deque, defaultdict
import warnings
import psutil
import gc
import easyocr

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path



# Professional logging setup to provide clear, timestamped updates
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler('sports_analytics.log', mode='w'),
        logging.StreamHandler()
    ]
)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress MPS pin_memory warning on Apple Silicon
warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")
# Suppress other MPS-related warnings
warnings.filterwarnings("ignore", message=".*MPS.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

@dataclass
class SystemMetrics:
    """Track system resource usage"""
    cpu_percent: float
    memory_percent: float
    available_memory_gb: float

# --- 1. ENTERPRISE CONFIGURATION ---
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
    def load_from_yaml(cls, yaml_path: str = 'config.yaml') -> None:
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


# --- 2. DEEP LEARNING MODEL BLUEPRINTS ---
class SegmentationModel:
    """Blueprint for a real DL segmentation model, with an advanced placeholder."""
    def __init__(self, model_path: str):
        self.model = None
        self.device = None
        self.transform = None
        self._initialize_model(model_path)

    def _initialize_model(self, model_path: str) -> None:
        """Initialize the segmentation model with proper error handling."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            if os.path.exists(model_path):
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                logging.info(f"Successfully loaded segmentation model '{model_path}' on {self.device}")
                
                # Standard image transformations for vision models
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((256, 256), antialias=True),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
        except (FileNotFoundError, RuntimeError) as e:
            self.model = None
            logging.warning(f"Segmentation model loading failed: {e}")
            logging.info("Switching to ADVANCED PLACEHOLDER mode (green pitch detection).")
        except Exception as e:
            self.model = None
            logging.error(f"Unexpected error loading segmentation model: {e}", exc_info=True)

    @contextmanager
    def _torch_inference_context(self):
        """Context manager for torch inference with proper cleanup"""
        try:
            torch.set_grad_enabled(False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Runs inference with the DL model or falls back to the placeholder."""
        if self.model and self.transform:
            try:
                return self._model_predict(frame)
            except Exception as e:
                logging.warning(f"Model prediction failed, falling back to placeholder: {e}")
                return self._placeholder_predict(frame)
        else:
            return self._placeholder_predict(frame)

    def _model_predict(self, frame: np.ndarray) -> np.ndarray:
        """Real DL Inference Workflow with improved memory management."""
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame")
            
        try:
            with self._torch_inference_context():
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.model is None:
                    raise RuntimeError("Segmentation model is not loaded")
                if self.transform is None:
                    raise RuntimeError("Transform is not initialized")
                
                # Add input validation
                if rgb_frame.shape[0] < 64 or rgb_frame.shape[1] < 64:
                    logging.warning("Input frame too small for segmentation")
                    return self._placeholder_predict(frame)
                
                input_tensor = self.transform(rgb_frame)
                if not isinstance(input_tensor, torch.Tensor):
                    input_tensor = torch.from_numpy(input_tensor)
                input_tensor = input_tensor.unsqueeze(0).to(self.device)
                
                # Add timeout for inference
                start_time = time.time()
                output = self.model(input_tensor)['out'][0]
                inference_time = time.time() - start_time
                
                if inference_time > 5.0:  # 5 second timeout
                    logging.warning(f"Slow inference: {inference_time:.2f}s")
                
                mask = torch.argmax(output, dim=0).detach().cpu().numpy().astype(np.uint8)
                # Resize mask back to original frame size
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                # Assume class '1' corresponds to pitch lines in the trained model
                line_mask = np.where(mask_resized == 1, 255, 0).astype(np.uint8)
                return line_mask
                
        except torch.cuda.OutOfMemoryError:
            logging.error("CUDA out of memory during segmentation")
            torch.cuda.empty_cache()
            return self._placeholder_predict(frame)
        except Exception as e:
            logging.error(f"Model prediction failed: {e}")
            return self._placeholder_predict(frame)

    def _placeholder_predict(self, frame: np.ndarray) -> np.ndarray:
        """Optimized placeholder that finds lines on the green parts of the pitch with reduced computation."""
        try:
            # Check for invalid frame
            if frame is None or frame.size == 0 or len(frame.shape) != 3:
                return np.zeros((0, 0), dtype=np.uint8)

            # Performance optimization: Resize frame to reduce computation
            height, width = frame.shape[:2]
            if height > 480 or width > 640:
                scale_factor = min(640/width, 480/height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                frame_resized = cv2.resize(frame, (new_width, new_height))
            else:
                frame_resized = frame
                scale_factor = 1.0

            hsv_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

            # Simplified green detection with single range for performance
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])

            pitch_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

            # Simplified morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Smaller kernel
            pitch_mask = cv2.morphologyEx(pitch_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Skip the bitwise_and operation and work directly on grayscale
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

            # Apply mask directly to grayscale
            gray_masked = cv2.bitwise_and(gray_frame, pitch_mask)

            # Skip Gaussian blur for performance
            line_mask = cv2.Canny(gray_masked, 50, 150, apertureSize=3)

            # Scale back to original size if needed
            if scale_factor != 1.0:
                line_mask = cv2.resize(line_mask, (width, height), interpolation=cv2.INTER_NEAREST)

            return line_mask

        except Exception as e:
            logging.error(f"Placeholder prediction failed: {e}")
            # Return appropriate empty array based on frame validity
            if frame is not None and len(frame.shape) >= 2:
                return np.zeros(frame.shape[:2], dtype=np.uint8)
            else:
                return np.zeros((0, 0), dtype=np.uint8)


# --- 3. OBJECT TRACKER ---
class ObjectTracker:
    """Encapsulates the YOLO object tracker with improved error handling."""
    def __init__(self, model_path: str):
        self.model = None
        self.tracking_history = defaultdict(list)
        self.lost_track_threshold = 30  # frames
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: str) -> None:
        """Initialize YOLO model with better error handling"""
        try:
            if not os.path.exists(model_path):
                logging.warning(f"Model file not found: {model_path}, downloading...")
                # YOLO will auto-download if file doesn't exist

            # Suppress pin_memory warnings during YOLO initialization
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*pin_memory.*")
                warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")
                self.model = YOLO(model_path)
                # Test the model with a dummy frame
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model.predict(dummy_frame, verbose=False)

            logging.info("Object tracker initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize YOLO model: {e}")
            raise

    def track_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Enhanced object tracking with validation and history"""
        if frame is None or frame.size == 0:
            logging.warning("Invalid frame for tracking")
            return []
            
        try:
            # Add frame preprocessing for better detection
            if frame.shape[0] > 1080 or frame.shape[1] > 1920:
                scale_factor = min(1920/frame.shape[1], 1080/frame.shape[0])
                new_width = int(frame.shape[1] * scale_factor)
                new_height = int(frame.shape[0] * scale_factor)
                frame_resized = cv2.resize(frame, (new_width, new_height))
            else:
                frame_resized = frame
                scale_factor = 1.0
            
            objects = []
            if self.model is not None:
                # Suppress pin_memory warnings during YOLO tracking
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*pin_memory.*")
                    warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")
                    results = self.model.track(frame_resized, persist=True,
                                             classes=[0, 32], verbose=False,
                                             conf=0.3, iou=0.5)
                
                if results and results[0].boxes is not None:
                    # Ensure we have numpy arrays for iteration
                    def to_numpy(x):
                        if hasattr(x, 'cpu'):
                            arr = x.cpu().numpy()
                        else:
                            arr = np.array(x)
                        # Ensure array is at least 1D for iteration
                        if arr.ndim == 0:
                            arr = arr.reshape(1)
                        return arr

                    xyxy = to_numpy(results[0].boxes.xyxy)
                    ids = to_numpy(results[0].boxes.id) if results[0].boxes.id is not None else np.array([])
                    clss = to_numpy(results[0].boxes.cls)
                    confs = to_numpy(results[0].boxes.conf)

                    # Handle case where we have detections but no tracking IDs
                    if len(ids) == 0 and len(xyxy) > 0:
                        # Generate temporary IDs for detections without tracking
                        ids = np.arange(len(xyxy)) + 10000  # Use high numbers to avoid conflicts

                    # Ensure all arrays have the same length
                    min_len = min(len(xyxy), len(ids), len(clss), len(confs))
                    if min_len == 0:
                        return objects

                    xyxy = xyxy[:min_len]
                    ids = ids[:min_len]
                    clss = clss[:min_len]
                    confs = confs[:min_len]

                    for box, track_id, cls_id, conf in zip(xyxy, ids, clss, confs):
                        if conf < 0.3:
                            continue
                        
                        # Scale coordinates back if frame was resized
                        if scale_factor != 1.0:
                            box = box / scale_factor
                        
                        track_id_int = int(track_id)
                        
                        # Update tracking history
                        self.tracking_history[track_id_int].append({
                            'bbox': box.tolist(),
                            'confidence': float(conf),
                            'timestamp': time.time()
                        })
                        
                        # Keep only recent history
                        if len(self.tracking_history[track_id_int]) > 10:
                            self.tracking_history[track_id_int] = \
                                self.tracking_history[track_id_int][-10:]
                        
                        objects.append({
                            "id": track_id_int,
                            "type": self.model.names[int(cls_id)],
                            "bbox_video": [int(coord) for coord in box],
                            "confidence": float(conf),
                            "tracking_quality": self._calculate_tracking_quality(track_id_int)
                        })
            else:
                logging.error("YOLO model is not initialized. Cannot perform object tracking.")
            
            return objects
            
        except Exception as e:
            logging.error(f"Object tracking failed: {e}")
            return []
    
    def _calculate_tracking_quality(self, track_id: int) -> float:
        """Calculate tracking quality based on history"""
        history = self.tracking_history.get(track_id, [])
        if len(history) < 2:
            return 0.5
        
        # Calculate confidence stability
        confidences = [h['confidence'] for h in history[-5:]]
        conf_std = np.std(confidences)
        
        # Calculate position stability
        bboxes = [h['bbox'] for h in history[-3:]]
        if len(bboxes) >= 2:
            movement = np.linalg.norm(
                np.array(bboxes[-1][:2]) - np.array(bboxes[-2][:2])
            )
        else:
            movement = 0
        
        # Combine metrics (lower is better for both)
        quality = max(0.0, min(1.0, 1.0 - (conf_std + movement/100)))
        return float(quality)

# --- 4. HOMOGRAPHY MANAGER ---
class HomographyManager:
    """Manages homography with correct reprojection error calculation."""
    def __init__(self, pitch_dims: Tuple[int, int], segmentation_model: SegmentationModel):
        self.pitch_dims = pitch_dims
        self.segmentation_model = segmentation_model
        self.homography_matrix = None
        self.template_points = np.array([
            [0, 0], [pitch_dims[0], 0],
            [pitch_dims[0], pitch_dims[1]], [0, pitch_dims[1]]
        ], dtype=np.float32)
        self.last_src_points = None
        self.calibration_attempts = 0
        self.max_calibration_attempts = 5
        self.last_calibration_time = 0
        self.calibration_cooldown = 30  # seconds between calibration resets
        self.consecutive_failures = 0
        self.max_consecutive_failures = 20  # Stop trying after this many consecutive failures

    def calculate_homography(self, frame: np.ndarray) -> bool:
        """Calculates homography using the segmentation mask with improved robustness."""
        current_time = time.time()

        # Reset calibration attempts after cooldown period
        if current_time - self.last_calibration_time > self.calibration_cooldown:
            self.calibration_attempts = 0
            self.last_calibration_time = current_time

        # Check if we've had too many consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            # Stop trying for a very long time
            if current_time - self.last_calibration_time > self.calibration_cooldown * 10:
                logging.info("Long cooldown period reached after many failures, resetting calibration state")
                self.calibration_attempts = 0
                self.consecutive_failures = 0
                self.last_calibration_time = current_time
            else:
                return False

        if self.calibration_attempts >= self.max_calibration_attempts:
            # Instead of just warning and returning False, try a longer cooldown
            if current_time - self.last_calibration_time > self.calibration_cooldown * 3:
                logging.info("Extended cooldown period reached, resetting calibration attempts")
                self.calibration_attempts = 0
                self.last_calibration_time = current_time
            else:
                # Reduce frequency of warning messages - only warn once per cooldown period
                if (self.calibration_attempts == self.max_calibration_attempts and
                    current_time - self.last_calibration_time < 1.0):  # Only warn in first second
                    logging.warning("Max calibration attempts reached - will retry after extended cooldown")
                return False

        self.calibration_attempts += 1

        try:
            # Validate input frame
            if frame is None or frame.size == 0 or len(frame.shape) != 3:
                logging.warning("Invalid frame for homography calculation")
                return False

            line_mask = self.segmentation_model.predict(frame)
            if line_mask is None or line_mask.size == 0:
                logging.debug("No line mask generated from segmentation")
                return False

            # Improved line detection with multiple parameter sets
            lines = self._detect_lines_robust(line_mask)
            if lines is None or len(lines) < 4:
                logging.debug(f"Insufficient lines detected: {len(lines) if lines is not None else 0}")
                return False

            # Separate horizontal and vertical lines with better angle thresholds
            h_lines, v_lines = self._separate_lines(lines)

            if len(h_lines) < 2 or len(v_lines) < 2:
                logging.debug(f"Insufficient line separation: h={len(h_lines)}, v={len(v_lines)}")
                return False

            # Find pitch boundaries more robustly
            pitch_corners = self._find_pitch_corners(h_lines, v_lines)
            if pitch_corners is None:
                logging.debug("Could not find valid pitch corners")
                return False

            self.last_src_points = np.array(pitch_corners, dtype=np.float32)

            # Validate corner points before homography calculation
            if not self._validate_corner_points(self.last_src_points):
                logging.debug("Corner points failed validation")
                return False

            self.homography_matrix, mask = cv2.findHomography(
                self.last_src_points,
                self.template_points,
                cv2.RANSAC,
                5.0,
                maxIters=2000,
                confidence=0.995
            )

            if self.homography_matrix is not None and mask is not None:
                # Additional validation of homography quality
                if self._validate_homography_matrix():
                    logging.info("Homography calibrated successfully.")
                    self.calibration_attempts = 0  # Reset on success
                    self.consecutive_failures = 0  # Reset consecutive failures on success
                    return True
                else:
                    logging.debug("Homography matrix failed quality validation")
                    self.homography_matrix = None

        except Exception as e:
            logging.error(f"Homography calculation failed: {e}")

        # Increment consecutive failures
        self.consecutive_failures += 1
        return False

    def _detect_lines_robust(self, line_mask: np.ndarray) -> Optional[np.ndarray]:
        """Detect lines with multiple parameter sets for robustness."""
        if line_mask is None or line_mask.size == 0:
            return None

        # Apply additional preprocessing to improve line detection
        try:
            # Dilate to connect broken lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            line_mask = cv2.dilate(line_mask, kernel, iterations=1)

            # Apply Gaussian blur to reduce noise
            line_mask = cv2.GaussianBlur(line_mask, (3, 3), 0)

            param_sets = [
                (15, 15, 30),  # Very lenient first - lower threshold, shorter lines, bigger gaps
                (20, 20, 25),  # Very lenient
                (25, 25, 20),  # Lenient
                (30, 30, 15),  # threshold, minLineLength, maxLineGap - more lenient
                (40, 25, 25),  # balanced
                (50, 50, 10),
                (70, 70, 5),   # strict
            ]

            backup_lines = None
            for threshold, min_length, max_gap in param_sets:
                try:
                    lines = cv2.HoughLinesP(
                        line_mask, 1, np.pi / 180,
                        threshold=threshold,
                        minLineLength=min_length,
                        maxLineGap=max_gap
                    )
                    if lines is not None and len(lines) >= 4:
                        logging.debug(f"Found {len(lines)} lines with params: {threshold}, {min_length}, {max_gap}")
                        return lines
                    elif lines is not None and len(lines) >= 2 and backup_lines is None:
                        # Accept even 2 lines if we're struggling
                        logging.debug(f"Found only {len(lines)} lines with params: {threshold}, {min_length}, {max_gap} - keeping as backup")
                        # Continue to try other parameters, but keep this as backup
                        backup_lines = lines
                except cv2.error as e:
                    logging.debug(f"HoughLinesP failed with params {threshold}, {min_length}, {max_gap}: {e}")
                    continue

            # If we couldn't find 4+ lines but found some lines, return what we have
            if backup_lines is not None:
                logging.debug(f"Returning backup lines: {len(backup_lines)} lines found")
                return backup_lines

        except Exception as e:
            logging.error(f"Line detection preprocessing failed: {e}")

        return None

    def _separate_lines(self, lines: np.ndarray) -> Tuple[List, List]:
        """Separate lines into horizontal and vertical with improved angle detection."""
        h_lines, v_lines = [], []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip very short lines
            if abs(x2 - x1) < 10 and abs(y2 - y1) < 10:
                continue
                
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            
            # More flexible angle thresholds
            if abs(angle) < 20 or abs(angle - 180) < 20:
                h_lines.append(line)
            elif abs(abs(angle) - 90) < 20:
                v_lines.append(line)
                
        return h_lines, v_lines

    def _find_pitch_corners(self, h_lines: List, v_lines: List) -> Optional[List]:
        """Find pitch corners with better error handling."""
        try:
            # Check if we have enough lines
            if len(h_lines) < 2 or len(v_lines) < 2:
                return None

            # Sort lines by position
            h_lines.sort(key=lambda line: (line[0][1] + line[0][3]) / 2)
            v_lines.sort(key=lambda line: (line[0][0] + line[0][2]) / 2)

            top_line = h_lines[0][0]
            bottom_line = h_lines[-1][0]
            left_line = v_lines[0][0]
            right_line = v_lines[-1][0]

            # Calculate intersections
            tl = self._line_intersection(top_line, left_line)
            tr = self._line_intersection(top_line, right_line)
            bl = self._line_intersection(bottom_line, left_line)
            br = self._line_intersection(bottom_line, right_line)

            corners = [tl, tr, br, bl]
            
            # Validate all corners exist and form a reasonable quadrilateral
            if not all(corners):
                return None
                
            # Check if corners form a reasonable quadrilateral
            if not self._validate_quadrilateral(corners):
                return None
                
            return corners
            
        except Exception as e:
            logging.error(f"Corner detection failed: {e}")
            return None

    def _line_intersection(self, l1: np.ndarray, l2: np.ndarray) -> Optional[List[int]]:
        """Calculate intersection point of two lines with better precision."""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-10:  # Lines are parallel
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        px = int(x1 + t * (x2 - x1))
        py = int(y1 + t * (y2 - y1))
        
        return [px, py]

    def _validate_quadrilateral(self, corners: List) -> bool:
        """Validate that corners form a reasonable quadrilateral."""
        try:
            # Check if points are roughly in the right order and not too close
            for i in range(4):
                for j in range(i + 1, 4):
                    dist = np.linalg.norm(np.array(corners[i]) - np.array(corners[j]))
                    if dist < 50:  # Points too close
                        return False
                        
            # Check if quadrilateral area is reasonable
            area = cv2.contourArea(np.array(corners, dtype=np.float32))
            if area < 10000:  # Too small
                return False
                
            return True
            
        except Exception:
            return False

    def _validate_corner_points(self, corners: np.ndarray) -> bool:
        """Validate corner points for homography calculation."""
        try:
            if corners.shape != (4, 2):
                return False

            # Check if points are within reasonable bounds
            for corner in corners:
                if corner[0] < 0 or corner[1] < 0 or corner[0] > 4096 or corner[1] > 2160:
                    return False

            # Check if corners form a reasonable quadrilateral
            area = cv2.contourArea(corners.astype(np.float32))
            if area < 10000:  # Too small
                return False

            # Check if corners are not too close to each other
            for i in range(4):
                for j in range(i + 1, 4):
                    dist = np.linalg.norm(corners[i] - corners[j])
                    if dist < 50:  # Points too close
                        return False

            return True

        except Exception:
            return False

    def _validate_homography_matrix(self) -> bool:
        """Validate the quality of the homography matrix."""
        try:
            if self.homography_matrix is None:
                return False

            # Check if matrix is well-conditioned
            cond_number = np.linalg.cond(self.homography_matrix)
            if cond_number > 1e10:  # Poorly conditioned matrix
                return False

            # Check determinant to ensure it's not degenerate
            det = np.linalg.det(self.homography_matrix[:2, :2])
            if abs(det) < 1e-6:
                return False

            # Check reprojection error
            error = self.calculate_reprojection_error()
            if error > 50.0:  # Too high error
                return False

            return True

        except Exception:
            return False

    def calculate_reprojection_error(self) -> float:
        """Calculates the true reprojection error to monitor homography quality."""
        if self.homography_matrix is None or self.last_src_points is None:
            return float('inf')
        
        try:
            # More robust error calculation
            inv_homography = np.linalg.inv(self.homography_matrix)
            reprojected_points = cv2.perspectiveTransform(
                self.template_points.reshape(-1, 1, 2), 
                inv_homography
            )
            
            # Calculate both forward and backward errors
            forward_error = np.mean(np.linalg.norm(
                self.last_src_points - reprojected_points.reshape(4, 2), 
                axis=1
            ))
            
            # Also check if matrix is well-conditioned
            cond_number = np.linalg.cond(self.homography_matrix)
            if cond_number > 1e12:  # Poorly conditioned matrix
                return float('inf')
                
            return forward_error
            
        except (np.linalg.LinAlgError, cv2.error) as e:
            logging.warning(f"Error calculation failed: {e}")
            return float('inf')
            
    def apply_homography(self, tracked_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transforms object coordinates from video space to pitch space."""
        if self.homography_matrix is None:
            # Fallback: use simple scaling transformation
            return self._apply_fallback_transformation(tracked_objects)

        try:
            for obj in tracked_objects:
                x1, _, x2, y2 = obj['bbox_video']  # Fixed: Use underscore for unused y1 variable

                # Use bottom center of bounding box
                bottom_center = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)

                transformed_point = cv2.perspectiveTransform(
                    bottom_center.reshape(-1, 1, 2),
                    self.homography_matrix
                )

                obj['pos_pitch'] = transformed_point.flatten().tolist()

        except Exception as e:
            logging.error(f"Homography application failed: {e}")
            # Fallback to simple transformation
            return self._apply_fallback_transformation(tracked_objects)

        return tracked_objects

    def _apply_fallback_transformation(self, tracked_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply a simple scaling transformation when homography is not available."""
        try:
            # Assume a typical video resolution and scale to pitch dimensions
            video_width, video_height = 1920, 1080  # Default assumption
            pitch_width, pitch_height = self.pitch_dims

            for obj in tracked_objects:
                x1, _, x2, y2 = obj['bbox_video']

                # Use bottom center of bounding box
                center_x = (x1 + x2) / 2

                # Simple linear scaling
                pitch_x = (center_x / video_width) * pitch_width
                pitch_y = (y2 / video_height) * pitch_height

                obj['pos_pitch'] = [float(pitch_x), float(pitch_y)]

        except Exception as e:
            logging.error(f"Fallback transformation failed: {e}")
            # Set default positions
            for obj in tracked_objects:
                obj['pos_pitch'] = [0.0, 0.0]

        return tracked_objects


# --- 5. TEAM IDENTIFIER ---
class TeamIdentifier:
    """Efficiently collects crops and automatically assigns Team A/Team B labels."""
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        self.team_map: Optional[Dict[int, str]] = None
        self.is_fitted = False
        self.feature_samples: List[np.ndarray] = []
        self.example_crops: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.crop_samples: List[np.ndarray] = []
        self._lock = threading.Lock()  # Thread safety

    def _extract_player_features(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Creates a feature vector (color histogram) from a player's torso."""
        try:
            x1, y1, x2, y2 = bbox
            h, w = y2 - y1, x2 - x1
            
            # Validate crop dimensions
            if h < 20 or w < 20:
                return None
                
            # Extract torso region with bounds checking
            torso_y1 = max(0, y1 + int(h * 0.2))
            torso_y2 = min(frame.shape[0], y1 + int(h * 0.7))
            torso_x1 = max(0, x1 + int(w * 0.2))
            torso_x2 = min(frame.shape[1], x1 + int(w * 0.8))
            
            crop_bgr = frame[torso_y1:torso_y2, torso_x1:torso_x2]
            
            if crop_bgr.size < Config.MODEL_PARAMS['MIN_CROP_SIZE']:
                return None
            
            # Convert to HSV for better color representation
            crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            
            # Create more robust histogram
            bins = [Config.MODEL_PARAMS['TEAM_FEATURE_BINS']] * 2
            hist = cv2.calcHist([crop_hsv], [0, 1], None, bins, [0, 180, 0, 256])
            
            # Normalize and add small epsilon to avoid zeros
            cv2.normalize(hist, hist)
            hist = hist.flatten()
            hist = hist + 1e-7
            
            # L2 normalize
            hist = hist / np.linalg.norm(hist)
            
            return hist
            
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
            return None

    def collect_and_store_sample(self, frame: np.ndarray, bbox: List[int]) -> None:
        """Collects feature vector and corresponding crop image with thread safety and rate limiting."""
        with self._lock:
            # Performance optimization: Limit sample collection frequency
            # Only collect samples every few frames to reduce computational load
            current_time = time.time()
            if hasattr(self, 'last_sample_time'):
                if current_time - self.last_sample_time < 0.5:  # 0.5 second cooldown
                    return
            self.last_sample_time = current_time

            features = self._extract_player_features(frame, bbox)
            if features is None:
                return

            self.feature_samples.append(features)

            # Store crop for HITL display
            try:
                x1, y1, x2, y2 = bbox
                h, w = y2 - y1, x2 - x1

                crop_y1 = max(0, y1 + int(h * 0.2))
                crop_y2 = min(frame.shape[0], y1 + int(h * 0.7))
                crop_x1 = max(0, x1 + int(w * 0.2))
                crop_x2 = min(frame.shape[1], x1 + int(w * 0.8))

                crop_bgr = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                if crop_bgr.size > 0:
                    crop_resized = cv2.resize(crop_bgr, (64, 64))
                    self.crop_samples.append(crop_resized)
                    # If already fitted, update example crops
                    if self.is_fitted:
                        cluster_id = self.kmeans.predict([features])[0]
                        max_crops = Config.MODEL_PARAMS['MAX_CROPS_PER_CLUSTER']
                        if len(self.example_crops[cluster_id]) < max_crops:
                            self.example_crops[cluster_id].append(crop_resized)
                else:
                    self.crop_samples.append(np.zeros((64, 64, 3), dtype=np.uint8))

            except Exception as e:
                logging.error(f"Crop storage failed: {e}")

    def fit(self) -> None:
        """Fits the K-Means model and organizes crops by cluster."""
        if len(self.feature_samples) < self.n_clusters:
            logging.warning(f"Not enough samples ({len(self.feature_samples)}) for {self.n_clusters} clusters")
            return
            
        try:
            logging.info("Fitting team feature model...")
            self.kmeans.fit(self.feature_samples)
            self.is_fitted = True
            
            # Clear existing example crops
            self.example_crops.clear()
            
            # Organize existing crops by cluster
            for i, features in enumerate(self.feature_samples):
                if i < len(self.crop_samples):
                    cluster_id = self.kmeans.predict([features])[0]
                    max_crops = Config.MODEL_PARAMS['MAX_CROPS_PER_CLUSTER']
                    if len(self.example_crops[cluster_id]) < max_crops:
                        self.example_crops[cluster_id].append(self.crop_samples[i])
                        
            logging.info(f"Team K-Means model fitted with {len(self.feature_samples)} samples.")
            
        except Exception as e:
            logging.error(f"Model fitting failed: {e}")
            self.is_fitted = False

    def classify_player(self, frame: np.ndarray, bbox: List[int]) -> str:
        """Classifies a single player based on the fitted model and labeled map."""
        if not self.is_fitted:
            return "Unknown"
        
        try:
            features = self._extract_player_features(frame, bbox)
            if features is None:
                return "Unknown"
            
            pred_cluster = self.kmeans.predict([features])[0]
            
            if self.team_map:
                return self.team_map.get(pred_cluster, f"Cluster {pred_cluster}")
            else:
                return f"Cluster {pred_cluster}"
                
        except Exception as e:
            logging.error(f"Player classification failed: {e}")
            return "Unknown"


# --- 5.1. JERSEY NUMBER DETECTOR ---
class JerseyNumberDetector:
    """Detects jersey numbers from player crops using YOLO and OCR."""

    def __init__(self, yolo_model_path: Optional[str] = None):
        self.yolo_model = None
        self.ocr_reader = None
        self.jersey_cache = {}  # Cache detected jersey numbers by object_id
        self.confidence_threshold = 0.5
        self.ocr_confidence_threshold = 0.6
        self.last_ocr_times = {}  # Track last OCR time per object for rate limiting
        self._initialize_models(yolo_model_path)

    def _initialize_models(self, yolo_model_path: Optional[str] = None):
        """Initialize YOLO model and OCR reader for jersey number detection."""
        try:
            # Initialize YOLO model for number detection
            # You can use a custom trained model or a general text detection model
            if yolo_model_path and os.path.exists(yolo_model_path):
                # Suppress pin_memory warnings during YOLO initialization
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*pin_memory.*")
                    warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")
                    self.yolo_model = YOLO(yolo_model_path)
                logging.info(f"Loaded custom YOLO model for jersey detection: {yolo_model_path}")
            else:
                # Fallback to general object detection model
                # In practice, you'd want a model specifically trained for jersey numbers
                logging.info("Using general YOLO model - consider training a specific jersey number detection model")

            # Initialize EasyOCR for number recognition with optimized settings
            try:
                # Try to initialize EasyOCR with SSL certificate handling
                import ssl
                import certifi

                # Set SSL certificate environment variable
                os.environ['SSL_CERT_FILE'] = certifi.where()
                os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

                # Set up SSL context with proper certificates
                ssl_context = ssl.create_default_context(cafile=certifi.where())
                ssl._create_default_https_context = lambda: ssl_context

                # Initialize EasyOCR with optimized settings for performance
                # Disable GPU for MPS compatibility and use CPU with optimized settings
                self.ocr_reader = easyocr.Reader(
                    ['en'],
                    gpu=False,  # Disable GPU to avoid MPS issues
                    verbose=False,  # Reduce logging
                    download_enabled=True
                )
                logging.info("EasyOCR initialized successfully with SSL certificate handling")

            except Exception as e:
                logging.warning(f"Failed to initialize EasyOCR with SSL handling: {e}")
                try:
                    # Fallback: try without SSL verification (less secure but functional)
                    import ssl
                    ssl._create_default_https_context = ssl._create_unverified_context
                    self.ocr_reader = easyocr.Reader(
                        ['en'],
                        gpu=False,  # Disable GPU to avoid MPS issues
                        verbose=False,
                        download_enabled=True
                    )
                    logging.info("EasyOCR initialized successfully with unverified SSL context (fallback)")
                except Exception as e2:
                    logging.warning(f"Failed to initialize EasyOCR even with fallback: {e2}")
                    self.ocr_reader = None

        except Exception as e:
            logging.error(f"Failed to initialize jersey number detection: {e}")

    def detect_jersey_number(self, frame: np.ndarray, bbox: List[int], object_id: int) -> int:
        """
        Detect jersey number from player crop with optimized caching and frame skipping.

        Args:
            frame: Full video frame
            bbox: Player bounding box [x1, y1, x2, y2]
            object_id: Tracking ID of the player

        Returns:
            Jersey number (0 if not detected)
        """
        try:
            # Check cache first - return immediately if we have ANY cached result
            if object_id in self.jersey_cache:
                cached_number, confidence_count = self.jersey_cache[object_id]
                # Return cached number immediately if we have any confidence
                # Jersey numbers don't change during a game, so once detected, use cached value
                if confidence_count >= 1:
                    return cached_number

            # Performance optimization: Skip OCR if disabled or if we should limit frequency
            if not getattr(Config, 'ENABLE_OCR', True):
                return 0

            # Limit OCR frequency per object to reduce computational load
            # Only attempt OCR detection every 30 frames per object (1 second at 30fps)
            current_time = time.time()
            last_ocr_time_key = f"last_ocr_{object_id}"
            if hasattr(self, 'last_ocr_times'):
                if last_ocr_time_key in self.last_ocr_times:
                    if current_time - self.last_ocr_times[last_ocr_time_key] < 1.0:  # 1 second cooldown
                        return 0
            else:
                self.last_ocr_times = {}

            # Update last OCR time for this object
            self.last_ocr_times[last_ocr_time_key] = current_time

            # Extract player crop
            player_crop = self._extract_player_crop(frame, bbox)
            if player_crop is None:
                return 0

            # Focus on torso area where jersey numbers are typically located
            torso_crop = self._extract_torso_region(player_crop)
            if torso_crop is None:
                return 0

            # Detect jersey number using OCR (now called much less frequently)
            jersey_number = self._detect_number_with_ocr(torso_crop)

            # Update cache with detected number
            if jersey_number > 0:
                self._update_jersey_cache(object_id, jersey_number)
                logging.info(f"Detected jersey number {jersey_number} for object {object_id}")
                return jersey_number

            # If no number detected, return cached number if available
            if object_id in self.jersey_cache:
                return self.jersey_cache[object_id][0]

            return 0

        except Exception as e:
            logging.warning(f"Jersey number detection failed for object {object_id}: {e}")
            return 0

    def _extract_player_crop(self, frame: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Extract player crop from frame with validation."""
        try:
            x1, y1, x2, y2 = bbox

            # Validate bbox
            if x1 >= x2 or y1 >= y2:
                return None

            # Ensure bbox is within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))

            crop = frame[y1:y2, x1:x2]

            # Validate crop size
            if crop.shape[0] < 50 or crop.shape[1] < 30:
                return None

            return crop

        except Exception as e:
            logging.warning(f"Player crop extraction failed: {e}")
            return None

    def _extract_torso_region(self, player_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract torso region where jersey numbers are typically located."""
        try:
            h, w = player_crop.shape[:2]

            # Focus on upper torso area (typically where numbers are)
            # Adjust these ratios based on typical jersey number placement
            torso_y1 = int(h * 0.15)  # Start from 15% down from top
            torso_y2 = int(h * 0.65)  # End at 65% down from top
            torso_x1 = int(w * 0.2)   # Start from 20% from left
            torso_x2 = int(w * 0.8)   # End at 80% from left

            torso_crop = player_crop[torso_y1:torso_y2, torso_x1:torso_x2]

            if torso_crop.shape[0] < 20 or torso_crop.shape[1] < 15:
                return None

            return torso_crop

        except Exception as e:
            logging.warning(f"Torso region extraction failed: {e}")
            return None

    def _detect_number_with_ocr(self, torso_crop: np.ndarray) -> int:
        """Detect jersey number using OCR."""
        try:
            if self.ocr_reader is None:
                return 0

            # Preprocess image for better OCR
            processed_crop = self._preprocess_for_ocr(torso_crop)

            # Run OCR
            results = self.ocr_reader.readtext(processed_crop)

            # Extract numbers from OCR results
            for (_, text, confidence) in results:  # bbox not needed for this implementation
                try:
                    conf_value = float(confidence)
                except (ValueError, TypeError):
                    conf_value = 0.0
                if conf_value > self.ocr_confidence_threshold:
                    # Extract numeric characters
                    numeric_text = ''.join(filter(str.isdigit, text))
                    if numeric_text:
                        number = int(numeric_text)
                        # Validate jersey number range (typically 1-99)
                        if 1 <= number <= 99:
                            logging.debug(f"Detected jersey number: {number} (confidence: {conf_value:.2f})")
                            return number

            return 0

        except Exception as e:
            logging.warning(f"OCR number detection failed: {e}")
            return 0

    def _preprocess_for_ocr(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess image to improve OCR accuracy with optimized performance."""
        try:
            # Convert to grayscale
            if len(crop.shape) == 3:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = crop.copy()

            # Optimized resize - use smaller scale factor to reduce computation
            scale_factor = 2  # Reduced from 3 to 2 for better performance
            height, width = gray.shape
            new_height, new_width = height * scale_factor, width * scale_factor
            resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LINEAR)  # Changed from CUBIC to LINEAR for speed

            # Simplified contrast enhancement - use adaptive threshold instead of CLAHE
            # CLAHE is computationally expensive, so we'll use a simpler approach

            # Apply simple contrast stretching
            min_val, max_val = np.min(resized), np.max(resized)
            if max_val > min_val:
                enhanced = ((resized - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                enhanced = resized

            # Skip Gaussian blur to save computation time
            # Apply adaptive threshold directly for better text detection
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Simplified morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # Smaller kernel
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            return cleaned

        except Exception as e:
            logging.warning(f"OCR preprocessing failed: {e}")
            return crop

    def _update_jersey_cache(self, object_id: int, jersey_number: int):
        """Update jersey number cache with confidence tracking."""
        try:
            if object_id in self.jersey_cache:
                cached_number, count = self.jersey_cache[object_id]
                if cached_number == jersey_number:
                    # Same number detected again, increase confidence
                    self.jersey_cache[object_id] = (jersey_number, count + 1)
                else:
                    # Different number detected, reset if new number has higher confidence
                    if count < 2:  # If cached number has low confidence, replace it
                        self.jersey_cache[object_id] = (jersey_number, 1)
            else:
                # First detection for this object
                self.jersey_cache[object_id] = (jersey_number, 1)

        except Exception as e:
            logging.warning(f"Jersey cache update failed: {e}")

    def get_cached_jersey_number(self, object_id: int) -> int:
        """Get cached jersey number for an object."""
        if object_id in self.jersey_cache:
            return self.jersey_cache[object_id][0]
        return 0

    def clear_cache(self):
        """Clear the jersey number cache."""
        self.jersey_cache.clear()


# --- 5.2. COMPREHENSIVE PLAYER DATABASE ---
class PlayerDatabase:
    """
    Comprehensive player identification system that combines jersey detection,
    team identification, and player naming for accurate sports analytics.
    """

    def __init__(self, jersey_detector: Optional['JerseyNumberDetector'], team_identifier: 'TeamIdentifier'):
        self.jersey_detector = jersey_detector
        self.team_identifier = team_identifier
        self.player_registry = {}  # object_id -> PlayerInfo
        self.team_rosters = {}     # team_name -> {jersey_number -> player_name}
        self.jersey_to_player = {} # team_name -> {jersey_number -> player_name}
        self._lock = threading.Lock()

        # Load known player databases (could be from external sources)
        self._load_player_databases()

    @dataclass
    class PlayerInfo:
        """Complete player information"""
        object_id: int
        jersey_number: int
        team_name: str
        player_name: str
        confidence_score: float
        detection_count: int
        last_seen_frame: int

        def update_confidence(self, new_detection: bool):
            """Update confidence based on consistent detections"""
            if new_detection:
                self.detection_count += 1
                self.confidence_score = min(1.0, self.confidence_score + 0.1)
            else:
                self.confidence_score = max(0.0, self.confidence_score - 0.05)

    def _load_player_databases(self):
        """Load known player databases from external sources"""
        try:
            # This could load from:
            # 1. CSV files with team rosters
            # 2. Sports APIs
            # 3. Manual configuration files
            # 4. Previous game databases

            # Example team rosters (in practice, load from external sources)
            self.team_rosters = {
                'Team A': {
                    1: 'Goalkeeper_A',
                    2: 'Defender_A_1',
                    3: 'Defender_A_2',
                    4: 'Defender_A_3',
                    5: 'Defender_A_4',
                    6: 'Midfielder_A_1',
                    7: 'Midfielder_A_2',
                    8: 'Midfielder_A_3',
                    9: 'Forward_A_1',
                    10: 'Forward_A_2',
                    11: 'Forward_A_3'
                },
                'Team B': {
                    1: 'Goalkeeper_B',
                    2: 'Defender_B_1',
                    3: 'Defender_B_2',
                    4: 'Defender_B_3',
                    5: 'Defender_B_4',
                    6: 'Midfielder_B_1',
                    7: 'Midfielder_B_2',
                    8: 'Midfielder_B_3',
                    9: 'Forward_B_1',
                    10: 'Forward_B_2',
                    11: 'Forward_B_3'
                },
                # Keep backward compatibility
                'Team_A': {
                    1: 'Goalkeeper_A',
                    2: 'Defender_A_1',
                    3: 'Defender_A_2',
                    4: 'Defender_A_3',
                    5: 'Defender_A_4',
                    6: 'Midfielder_A_1',
                    7: 'Midfielder_A_2',
                    8: 'Midfielder_A_3',
                    9: 'Forward_A_1',
                    10: 'Forward_A_2',
                    11: 'Forward_A_3'
                },
                'Team_B': {
                    1: 'Goalkeeper_B',
                    2: 'Defender_B_1',
                    3: 'Defender_B_2',
                    4: 'Defender_B_3',
                    5: 'Defender_B_4',
                    6: 'Midfielder_B_1',
                    7: 'Midfielder_B_2',
                    8: 'Midfielder_B_3',
                    9: 'Forward_B_1',
                    10: 'Forward_B_2',
                    11: 'Forward_B_3'
                }
            }

            # Create reverse mapping for quick lookup
            for team_name, roster in self.team_rosters.items():
                self.jersey_to_player[team_name] = roster.copy()

            logging.info(f"Loaded player databases for {len(self.team_rosters)} teams")

        except Exception as e:
            logging.warning(f"Failed to load player databases: {e}")
            self.team_rosters = {}
            self.jersey_to_player = {}

    def analyze_player(self, frame: np.ndarray, obj: Dict[str, Any], frame_id: int) -> Dict[str, Any]:
        """
        Comprehensive player analysis combining all detection methods with performance optimizations.

        Args:
            frame: Video frame
            obj: Object detection result
            frame_id: Current frame number

        Returns:
            Enhanced object with complete player information
        """
        try:
            with self._lock:
                object_id = obj.get('id', -1)
                bbox = obj.get('bbox_video', [0, 0, 0, 0])

                # Performance optimization: Check if we already have complete info for this player
                if object_id in self.player_registry:
                    existing_player = self.player_registry[object_id]
                    # If we have high confidence data, skip expensive analysis
                    if (existing_player.confidence_score > 0.8 and
                        existing_player.jersey_number > 0 and
                        existing_player.team_name != 'Unknown'):

                        # Update with cached information
                        obj.update({
                            'jersey_number': existing_player.jersey_number,
                            'team': existing_player.team_name,
                            'player_name': self._determine_player_name(existing_player),
                            'confidence_score': existing_player.confidence_score,
                            'detection_count': existing_player.detection_count
                        })

                        # Update last seen frame
                        existing_player.last_seen_frame = frame_id
                        return obj

                # Step 1: Detect jersey number (if enabled and not already cached with high confidence)
                if self.jersey_detector:
                    jersey_number = self.jersey_detector.detect_jersey_number(frame, bbox, object_id)
                else:
                    jersey_number = 0  # Default when jersey detection is disabled

                # Step 2: Identify team (skip if we already have team info with high confidence)
                if (object_id in self.player_registry and
                    self.player_registry[object_id].team_name != 'Unknown' and
                    self.player_registry[object_id].confidence_score > 0.5):
                    team_name = self.player_registry[object_id].team_name
                else:
                    team_name = self.team_identifier.classify_player(frame, bbox)

                # Step 3: Get or create player info
                player_info = self._get_or_create_player_info(object_id, jersey_number, team_name, frame_id)

                # Step 4: Determine player name
                player_name = self._determine_player_name(player_info)

                # Step 5: Update object with complete information
                obj.update({
                    'jersey_number': player_info.jersey_number,
                    'team': player_info.team_name,
                    'player_name': player_name,
                    'confidence_score': player_info.confidence_score,
                    'detection_count': player_info.detection_count
                })

                # Step 6: Update player info
                player_info.last_seen_frame = frame_id
                self.player_registry[object_id] = player_info

                return obj

        except Exception as e:
            logging.warning(f"Player analysis failed for object {obj.get('id', 'unknown')}: {e}")
            # Return object with fallback values
            obj.update({
                'jersey_number': 0,
                'team': 'Unknown',
                'player_name': f"Player_{obj.get('id', 0)}",
                'confidence_score': 0.0,
                'detection_count': 0
            })
            return obj

    def _get_or_create_player_info(self, object_id: int, jersey_number: int, team_name: str, frame_id: int) -> 'PlayerDatabase.PlayerInfo':
        """Get existing player info or create new one"""
        if object_id in self.player_registry:
            player_info = self.player_registry[object_id]

            # Update information if we have better detection
            updated = False
            if jersey_number > 0 and player_info.jersey_number != jersey_number:
                player_info.jersey_number = jersey_number
                updated = True

            if team_name != 'Unknown' and player_info.team_name != team_name:
                player_info.team_name = team_name
                updated = True

            player_info.update_confidence(updated)
            return player_info
        else:
            # Create new player info
            return self.PlayerInfo(
                object_id=object_id,
                jersey_number=jersey_number if jersey_number > 0 else 0,
                team_name=team_name if team_name != 'Unknown' else 'Unknown',
                player_name='',  # Will be determined later
                confidence_score=0.5,
                detection_count=1,
                last_seen_frame=frame_id
            )

    def _determine_player_name(self, player_info: 'PlayerDatabase.PlayerInfo') -> str:
        """Determine player name from jersey number and team"""
        try:
            team_name = player_info.team_name
            jersey_number = player_info.jersey_number

            # First, try to get name from known roster
            if (team_name in self.jersey_to_player and
                jersey_number in self.jersey_to_player[team_name]):
                return self.jersey_to_player[team_name][jersey_number]

            # If not in roster, generate descriptive name based on jersey number
            if jersey_number > 0:
                if jersey_number == 1:
                    return f"{team_name}_Goalkeeper"
                elif jersey_number <= 5:
                    return f"{team_name}_Defender_{jersey_number}"
                elif jersey_number <= 8:
                    return f"{team_name}_Midfielder_{jersey_number}"
                else:
                    return f"{team_name}_Forward_{jersey_number}"
            else:
                # No jersey number detected
                return f"{team_name}_Player_{player_info.object_id}"

        except Exception as e:
            logging.warning(f"Player name determination failed: {e}")
            return f"Player_{player_info.object_id}"

    def get_player_info(self, object_id: int) -> Optional['PlayerDatabase.PlayerInfo']:
        """Get player information by object ID"""
        return self.player_registry.get(object_id)

    def get_team_roster(self, team_name: str) -> Dict[int, str]:
        """Get complete roster for a team"""
        return self.team_rosters.get(team_name, {})



    def load_external_roster(self, roster_file: str):
        """Load team roster from external file (CSV, JSON, etc.)"""
        try:
            if roster_file.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(roster_file)
                # Expected columns: team_name, jersey_number, player_name
                for _, row in df.iterrows():
                    team = row['team_name']
                    jersey = int(row['jersey_number'])
                    name = row['player_name']

                    if team not in self.team_rosters:
                        self.team_rosters[team] = {}
                        self.jersey_to_player[team] = {}

                    self.team_rosters[team][jersey] = name
                    self.jersey_to_player[team][jersey] = name

                logging.info(f"Loaded external roster from {roster_file}")

        except Exception as e:
            logging.error(f"Failed to load external roster: {e}")





# --- 6. ACTION RECOGNIZER HARNESS ---
class ActionRecognizer:
    """Harness for a sequence-based action recognition model."""
    def __init__(self):
        self.sequence_model = None  # Placeholder for a loaded model
        self.feature_sequence = deque(maxlen=Config.ACTION_SEQUENCE_LENGTH)
        self.ball_positions = deque(maxlen=Config.ACTION_SEQUENCE_LENGTH)
        self.last_event_frame = 0
        self.event_cooldown = 30  # Frames between events
        logging.info("ActionRecognizer harness initialized.")

    def recognize(self, tracked_objects: List[Dict[str, Any]], frame_id: int) -> Dict[str, Any]:
        """Collects sequences and runs simple heuristic-based action recognition."""
        try:
            # Find ball
            ball = next((obj for obj in tracked_objects 
                        if obj.get('type') == 'sports ball' and obj.get('pos_pitch')), None)
            
            if ball:
                pos = ball['pos_pitch']
                self.ball_positions.append(pos)
                self.feature_sequence.append(pos)
            else:
                self.ball_positions.append(None)
                self.feature_sequence.append(None)
            
            # Simple event detection
            if frame_id - self.last_event_frame > self.event_cooldown:
                event = self._detect_simple_events(frame_id)
                if event:
                    self.last_event_frame = frame_id
                    return event
                    
        except Exception as e:
            logging.error(f"Action recognition failed: {e}")
            
        return {}

    def _detect_simple_events(self, frame_id: int) -> Dict[str, Any]:
        """Simple heuristic-based event detection."""
        try:
            valid_positions = [p for p in self.ball_positions if p is not None]
            
            if len(valid_positions) < 5:
                return {}
                
            # Calculate velocity and acceleration
            recent_positions = valid_positions[-5:]
            velocities = []
            
            for i in range(1, len(recent_positions)):
                vel = np.linalg.norm(np.array(recent_positions[i]) - np.array(recent_positions[i-1]))
                velocities.append(vel)
            
            if not velocities:
                return {}
                
            max_velocity = max(velocities)
            avg_velocity = np.mean(velocities)
            
            # Event detection thresholds
            if max_velocity > 50:
                return {"event": "Fast Ball Movement", "frame": frame_id, "velocity": max_velocity}
            elif avg_velocity > 20:
                return {"event": "Ball Movement", "frame": frame_id, "velocity": avg_velocity}
                
        except Exception as e:
            logging.error(f"Event detection failed: {e}")
            
        return {}


# --- 7. MAIN VIDEO PROCESSOR ORCHESTRATOR ---
class VideoProcessor:
    """The main orchestrator for the entire analytics pipeline."""

    def _reset_log_file(self):
        """Reset the log file for a new video processing session."""
        try:
            # Clear the sports_analytics.log file by opening it in write mode
            with open('sports_analytics.log', 'w'):
                pass  # Just opening in 'w' mode clears the file
            logging.info("Log file reset for new video processing session")
        except Exception as e:
            logging.warning(f"Failed to reset log file: {e}")

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
        self.dashboard_server = None
        if enable_streaming:
            try:
                from dashboard_server import dashboard_server
                self.dashboard_server = dashboard_server
                logging.info("Real-time streaming enabled")
            except ImportError:
                logging.warning("Dashboard server not available, streaming disabled")
                self.enable_streaming = False

        # Initialize video capture
        self._initialize_video()

        # Initialize components
        self._initialize_components()

        # Load existing state
        self.load_state()

        # Add performance monitoring
        self.performance_metrics = {
            'frames_processed': 0,
            'frames_skipped': 0,
            'processing_times': deque(maxlen=100),
            'memory_usage': deque(maxlen=50),
            'errors_count': 0
        }
        
        # Add graceful shutdown handling
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

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
                self.total_frames = 0
                self.width = 1920  # Default dimensions
                self.height = 1080
                self.video_hash = "runtime_upload"
                return

            # Try to open video with better error handling
            try:
                self.cap = cv2.VideoCapture(self.config.VIDEO_PATH)
                if not self.cap.isOpened():
                    raise IOError(f"Cannot open video file: {self.config.VIDEO_PATH}")

                # Get video properties with validation
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Validate video properties
                if self.fps <= 0:
                    logging.warning("Invalid FPS detected, using default value of 30")
                    self.fps = 30
                if self.total_frames <= 0:
                    logging.warning("Invalid frame count detected")
                if self.width <= 0 or self.height <= 0:
                    logging.warning(f"Invalid video dimensions: {self.width}x{self.height}")

                logging.info(f"Video: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")

                # Calculate robust video hash
                self.video_hash = self._calculate_robust_video_hash()
                self.config.generate_checkpoint_prefix(self.video_hash)

            except cv2.error as e:
                logging.error(f"OpenCV error opening video: {e}")
                raise IOError(f"OpenCV cannot process video file: {self.config.VIDEO_PATH}")
            except Exception as e:
                logging.error(f"Unexpected error opening video: {e}")
                raise

        except Exception as e:
            logging.error(f"Video initialization failed: {e}")
            raise

    def set_video_path(self, video_path: str) -> None:
        """Set video path and reinitialize video capture for runtime uploads."""
        try:
            # Close existing capture if any
            if self.cap:
                self.cap.release()

            # Update config and reinitialize
            self.config.VIDEO_PATH = video_path
            self._initialize_video()

            logging.info(f"Video path updated and reinitialized: {video_path}")

        except Exception as e:
            logging.error(f"Failed to set video path: {e}")
            raise

    def _initialize_components(self) -> None:
        """Initialize all processing components."""
        try:
            self.segmentation_model = SegmentationModel(self.config.SEGMENTATION_MODEL_PATH)
            self.homography_manager = HomographyManager(
                pitch_dims=(1050, 680),
                segmentation_model=self.segmentation_model
            )
            self.team_identifier = TeamIdentifier(self.config.MODEL_PARAMS['TEAM_N_CLUSTERS'])
            self.object_tracker = ObjectTracker(self.config.YOLO_MODEL_PATH)
            self.action_recognizer = ActionRecognizer()

            # Initialize jersey number detector (conditionally for performance)
            if getattr(self.config, 'ENABLE_JERSEY_DETECTION', True):
                jersey_model_path = getattr(self.config, 'JERSEY_YOLO_MODEL_PATH', None)
                self.jersey_detector = JerseyNumberDetector(jersey_model_path)
            else:
                self.jersey_detector = None
                logging.info("Jersey number detection disabled for performance optimization")

            # Initialize comprehensive player database
            self.player_database = PlayerDatabase(self.jersey_detector, self.team_identifier)

            # Load external roster if provided
            if hasattr(self.config, 'TEAM_ROSTER_PATH') and self.config.TEAM_ROSTER_PATH:
                self.player_database.load_external_roster(self.config.TEAM_ROSTER_PATH)



            logging.info("All components initialized successfully")

        except Exception as e:
            logging.error(f"Component initialization failed: {e}")
            raise

    def _calculate_robust_video_hash(self) -> str:
        """Hashes video file properties for robustness without reading frames."""
        logging.info("Calculating robust video hash...")

        try:
            hasher = hashlib.md5()

            # Use video file properties instead of reading frames to avoid OpenCV issues
            if self.config.VIDEO_PATH and os.path.exists(self.config.VIDEO_PATH):
                # Use file stats for hashing
                stat = os.stat(self.config.VIDEO_PATH)
                hasher.update(str(stat.st_size).encode())
                hasher.update(str(stat.st_mtime).encode())
                hasher.update(self.config.VIDEO_PATH.encode())

                # Add video properties if available
                if self.cap is not None:
                    try:
                        hasher.update(str(self.fps).encode())
                        hasher.update(str(self.total_frames).encode())
                        hasher.update(str(self.width).encode())
                        hasher.update(str(self.height).encode())
                    except Exception as e:
                        logging.debug(f"Could not add video properties to hash: {e}")

                return hasher.hexdigest()[:16]
            else:
                # Fallback for runtime uploads
                return hashlib.md5(str(time.time()).encode()).hexdigest()[:16]

        except Exception as e:
            logging.error(f"Video hash calculation failed: {e}")
            return hashlib.md5(str(time.time()).encode()).hexdigest()[:16]

    def load_state(self) -> None:
        """Loads learned models from configuration-aware checkpoints."""
        try:
            team_model_path = f"{self.config.checkpoint_path_prefix}_team_model.joblib"
            homography_path = f"{self.config.checkpoint_path_prefix}_homography.npy"
            
            if os.path.exists(team_model_path):
                try:
                    # Try to load as the new format (dictionary)
                    team_data = joblib.load(team_model_path)
                    if isinstance(team_data, dict):
                        # Restore team identifier from dictionary
                        self.team_identifier.n_clusters = team_data['n_clusters']
                        self.team_identifier.kmeans = team_data['kmeans']
                        self.team_identifier.team_map = team_data['team_map']
                        self.team_identifier.is_fitted = team_data['is_fitted']
                        self.team_identifier.feature_samples = team_data['feature_samples']
                        self.team_identifier.example_crops = defaultdict(list, team_data['example_crops'])
                        self.team_identifier.crop_samples = team_data['crop_samples']
                    else:
                        # Old format - try to load directly (may fail due to lock)
                        self.team_identifier = team_data
                    logging.info("Loaded team identification model from checkpoint.")
                except Exception as load_error:
                    logging.warning(f"Failed to load team model checkpoint: {load_error}")
                
            if os.path.exists(homography_path):
                self.homography_manager.homography_matrix = np.load(homography_path)
                logging.info("Loaded homography matrix from checkpoint.")
                
        except Exception as e:
            logging.error(f"State loading failed: {e}")

    def save_state(self) -> None:
        """Saves learned models to disk for future runs."""
        try:
            os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
            
            team_model_path = f"{self.config.checkpoint_path_prefix}_team_model.joblib"
            homography_path = f"{self.config.checkpoint_path_prefix}_homography.npy"
            
            if self.team_identifier.is_fitted and self.team_identifier.team_map:
                # Create a copy of team_identifier without the lock for pickling
                team_data = {
                    'n_clusters': self.team_identifier.n_clusters,
                    'kmeans': self.team_identifier.kmeans,
                    'team_map': self.team_identifier.team_map,
                    'is_fitted': self.team_identifier.is_fitted,
                    'feature_samples': self.team_identifier.feature_samples,
                    'example_crops': dict(self.team_identifier.example_crops),
                    'crop_samples': self.team_identifier.crop_samples
                }
                joblib.dump(team_data, team_model_path)
                logging.info(f"Saved team model to {team_model_path}")
                
            if self.homography_manager.homography_matrix is not None:
                np.save(homography_path, self.homography_manager.homography_matrix)
                logging.info(f"Saved homography matrix to {homography_path}")
                
        except Exception as e:
            logging.error(f"State saving failed: {e}")

    def _perform_visual_hitl(self) -> None:
        """Automatically assign Team A and Team B labels without human input."""
        try:
            logging.info("Automatically assigning team labels...")

            cluster_ids = sorted(self.team_identifier.example_crops.keys())

            if not cluster_ids:
                logging.error("No clusters found for team assignment.")
                return

            # Automatically assign Team A, Team B, etc.
            self._use_default_labels(cluster_ids)

        except Exception as e:
            logging.error(f"Automatic team assignment failed: {e}")
            # Fallback to default labeling
            cluster_ids = sorted(self.team_identifier.example_crops.keys())
            if cluster_ids:
                self._use_default_labels(cluster_ids)

    def _perform_text_hitl(self, cluster_ids: List[int]) -> None:
        """Automatically assign team labels without human input."""
        try:
            logging.info("Automatically assigning team labels...")

            # Always use default labels without human input
            self._use_default_labels(cluster_ids)

        except Exception as e:
            logging.error(f"Automatic team assignment failed: {e}")
            self._use_default_labels(cluster_ids)

    def _use_default_labels(self, cluster_ids: List[int]) -> None:
        """Use default labels with Team A and Team B naming."""
        try:
            # Use Team A, Team B, Team C, etc. for better readability
            team_names = ["Team A", "Team B", "Team C", "Team D", "Team E"]
            default_labels = [team_names[i] if i < len(team_names) else f"Team {chr(65+i)}" for i in range(len(cluster_ids))]
            self.team_identifier.team_map = {cluster_ids[i]: default_labels[i] for i in range(len(cluster_ids))}
            logging.info(f"Default team labels applied: {self.team_identifier.team_map}")
        except Exception as e:
            logging.error(f"Failed to apply default labels: {e}")



    def _processing_loop(self) -> None:
        """Enhanced processing loop with frame skipping and better error handling"""
        frame_id = 0
        retry_count = 0
        last_metrics_log = time.time()
        
        while not self.stop_event.is_set():
            start_time = time.time()
            
            try:
                frame_data = self.frame_queue.get(timeout=self.config.PROCESSING_TIMEOUT)
                frame, actual_frame_id = frame_data
                retry_count = 0
                
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                retry_count += 1
                if retry_count > self.config.MAX_RETRIES:
                    logging.warning("Maximum retries reached in processing loop")
                    break
                continue

            try:
                # Monitor system resources periodically
                if time.time() - last_metrics_log > 30:  # Every 30 seconds
                    metrics = Config.get_system_metrics()
                    logging.info(f"System: CPU {metrics.cpu_percent:.1f}%, "
                               f"Memory {metrics.memory_percent:.1f}%, "
                               f"Available {metrics.available_memory_gb:.1f}GB")
                    self.performance_metrics['memory_usage'].append(metrics.memory_percent)
                    last_metrics_log = time.time()
                
                # Check memory usage and clean up if needed
                if psutil.virtual_memory().percent > 85:
                    logging.warning("High memory usage, forcing garbage collection")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Object tracking with error recovery
                try:
                    objects = self.object_tracker.track_objects(frame)
                except Exception as e:
                    logging.error(f"Object tracking failed for frame {actual_frame_id}: {e}")
                    objects = []
                
                # Team identification workflow with better error handling
                if not self.team_identifier.is_fitted:
                    if len(self.team_identifier.feature_samples) < self.config.TEAM_SAMPLES_TO_COLLECT:
                        # Collect samples
                        for obj in objects:
                            if obj['type'] == 'person':
                                try:
                                    self.team_identifier.collect_and_store_sample(frame, obj['bbox_video'])
                                except Exception as e:
                                    logging.warning(f"Sample collection failed: {e}")
                    else:
                        # Fit model
                        try:
                            self.team_identifier.fit()
                            
                            # Perform team identification - always use automatic labeling
                            if not self.team_identifier.team_map:
                                # Always use automatic Team A/Team B labeling
                                cluster_ids = sorted(self.team_identifier.example_crops.keys())
                                if cluster_ids:
                                    self._use_default_labels(cluster_ids)
                        except Exception as e:
                            logging.error(f"Team model fitting failed: {e}")

                # Homography management with better error recovery
                if (self.config.ENABLE_HOMOGRAPHY):
                    # Much more aggressive rate limiting for homography calculation
                    # Only attempt calibration every 50 frames (2 seconds at 25fps) to reduce computational load
                    if (self.homography_manager.homography_matrix is None and
                        actual_frame_id % 50 == 0):  # Increased from 10 to 50 frames
                        try:
                            success = self.homography_manager.calculate_homography(frame)
                            if success:
                                logging.info(f"Homography successfully calibrated at frame {actual_frame_id}")
                        except Exception as e:
                            logging.error(f"Homography calculation failed: {e}")

                    # Much less frequent homography quality checks (every 10 seconds instead of every 6 seconds)
                    if (actual_frame_id > 0 and
                        actual_frame_id % (self.config.HOMOGRAPHY_CHECK_INTERVAL * 4) == 0 and  # Increased from 2x to 4x
                        self.homography_manager.homography_matrix is not None):
                        try:
                            error = self.homography_manager.calculate_reprojection_error()
                            if error > self.config.HOMOGRAPHY_RECAL_THRESHOLD:
                                logging.warning(f"High reprojection error ({error:.2f}px). Triggering recalibration.")
                                self.homography_manager.homography_matrix = None
                        except Exception as e:
                            logging.warning(f"Homography error check failed: {e}")
                else:
                    # Homography disabled - log once and ensure we use fallback
                    if actual_frame_id == 1:
                        logging.info("Homography calibration disabled in configuration - using fallback transformation")

                # Comprehensive player analysis (team, jersey, name all at once)
                for obj in objects:
                    if obj['type'] == 'person':
                        try:
                            # Use comprehensive player database for complete analysis
                            obj = self.player_database.analyze_player(frame, obj, actual_frame_id)
                        except Exception as e:
                            logging.warning(f"Comprehensive player analysis failed for object {obj.get('id', 'unknown')}: {e}")
                            # Fallback to basic values
                            obj.update({
                                'team': 'Unknown',
                                'jersey_number': 0,
                                'player_name': f"Player_{obj.get('id', 0)}",
                                'confidence_score': 0.0
                            })
                
                # Apply homography transformation
                try:
                    objects = self.homography_manager.apply_homography(objects)
                except Exception as e:
                    logging.warning(f"Homography application failed: {e}")
                
                # Action recognition
                try:
                    actions = self.action_recognizer.recognize(objects, actual_frame_id)
                except Exception as e:
                    logging.warning(f"Action recognition failed: {e}")
                    actions = {}

                # Store results with thread safety
                frame_result = {
                    "frame_id": actual_frame_id,
                    "timestamp": actual_frame_id / self.fps if self.fps > 0 else 0,
                    "actions": actions,
                    "objects": objects
                }

                with self.processing_lock:
                    self.results_data.append(frame_result)

                # Stream data to dashboard if enabled
                if self.enable_streaming and self.dashboard_server:
                    try:
                        # Use a simple queue-based approach to avoid async complexity
                        if hasattr(self.dashboard_server, 'data_queue'):
                            self.dashboard_server.data_queue.put_nowait(frame_result)
                        else:
                            # Fallback: store latest data for polling
                            self.dashboard_server.latest_data = frame_result
                            self.dashboard_server._update_game_stats(frame_result)
                    except Exception as e:
                        logging.debug(f"Streaming failed: {e}")  # Use debug level to avoid spam
                
                # Track performance metrics
                processing_time = time.time() - start_time
                self.performance_metrics['processing_times'].append(processing_time)
                self.performance_metrics['frames_processed'] += 1

                # Improved performance monitoring with different thresholds
                if processing_time > 2.0:  # Very slow frames
                    logging.warning(f"Slow processing: {processing_time:.2f}s for frame {actual_frame_id}")
                elif processing_time > 1.0:  # Moderately slow frames
                    logging.info(f"Moderate processing time: {processing_time:.2f}s for frame {actual_frame_id}")
                elif actual_frame_id % 100 == 0:  # Log every 100th frame for progress tracking
                    logging.info(f"Processing frame {actual_frame_id} in {processing_time:.2f}s")
                
                frame_id += 1
                
            except Exception as e:
                logging.error(f"Processing error at frame {actual_frame_id}: {e}")
                self.performance_metrics['errors_count'] += 1
                frame_id += 1
                
        # Log final performance metrics
        if self.performance_metrics['processing_times']:
            avg_time = np.mean(self.performance_metrics['processing_times'])
            logging.info(f"Average processing time: {avg_time:.3f}s per frame")
            logging.info(f"Total frames processed: {self.performance_metrics['frames_processed']}")
            logging.info(f"Total frames skipped: {self.performance_metrics['frames_skipped']}")
            logging.info(f"Total errors: {self.performance_metrics['errors_count']}")

    def start(self) -> None:
        """Starts the video reading and processing threads with frame skipping."""
        try:
            # Start processing thread
            processing_thread = threading.Thread(
                target=self._processing_loop, 
                name="ProcessingThread",
                daemon=True
            )
            processing_thread.start()
            
            # Main video reading loop with frame skipping
            frame_id = 0
            last_progress_log = 0
            
            while True:
                if self.stop_event.is_set() and self.frame_queue.empty():
                    break
                
                # Read frame if queue has space
                if not self.frame_queue.full():
                    if self.cap is not None:
                        try:
                            ret, frame = self.cap.read()
                            if not ret:
                                logging.info("End of video file reached.")
                                self.stop_event.set()
                                time.sleep(1)  # Give processing thread time to finish
                                break
                        except cv2.error as e:
                            logging.error(f"OpenCV error reading frame: {e}")
                            self.stop_event.set()
                            break
                        except Exception as e:
                            logging.error(f"Unexpected error reading frame: {e}")
                            self.stop_event.set()
                            break
                    else:
                        logging.error("VideoCapture object is None. Cannot read frames.")
                        self.stop_event.set()
                        break
                    
                    # Implement frame skipping for performance
                    self.frame_skip_counter += 1
                    if self.frame_skip_counter % self.config.MODEL_PARAMS['FRAME_SKIP_INTERVAL'] != 0:
                        self.performance_metrics['frames_skipped'] += 1
                        frame_id += 1
                        continue
                    
                    try:
                        self.frame_queue.put((frame, frame_id), timeout=0.1)
                        frame_id += 1
                        
                        # Progress logging
                        if frame_id - last_progress_log >= 100:
                            progress = (frame_id / self.total_frames) * 100 if self.total_frames > 0 else 0
                            logging.info(f"Read {frame_id} frames ({progress:.1f}%)")
                            last_progress_log = frame_id
                            
                    except queue.Full:
                        time.sleep(0.001)  # Brief pause if queue is full
                        
                else:
                    time.sleep(0.001)  # Brief pause if queue is full

                # Check for quit signal (only in non-headless mode)
                try:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logging.info("'q' pressed, shutting down.")
                        self.stop_event.set()
                        break
                except cv2.error:
                    # Ignore OpenCV errors in headless mode
                    pass
            
            # Wait for processing to complete
            logging.info("Waiting for processing thread to complete...")
            processing_thread.join(timeout=30)
            
            if processing_thread.is_alive():
                logging.warning("Processing thread did not complete within timeout")
                
        except Exception as e:
            logging.error(f"Error in main processing loop: {e}")
            self.stop_event.set()
            
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Saves state and exports data upon completion."""
        try:
            logging.info("Cleaning up, saving state, and exporting data...")
            
            # Release resources
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Save state
            self.save_state()



        except Exception as e:
            logging.error(f"Cleanup failed: {e}")


    def _assign_jersey_number(self, obj_id: int, team_name: str) -> int:
        """
        Assign jersey numbers to players using the jersey detector.
        Falls back to consistent ID-based assignment if detection fails.
        """
        try:
            # First check if we have a detected jersey number from the current frame
            if hasattr(self, 'jersey_detector') and self.jersey_detector is not None:
                detected_number = self.jersey_detector.get_cached_jersey_number(obj_id)
                if detected_number > 0:
                    return detected_number

            # If no detection, use consistent ID-based assignment as fallback
            if not hasattr(self, '_jersey_assignments'):
                self._jersey_assignments = {}

            team_key = f"{team_name}_{obj_id}"

            if team_key not in self._jersey_assignments:
                # Get existing jersey numbers for this team
                team_jerseys = set()
                for key, jersey in self._jersey_assignments.items():
                    if key.startswith(f"{team_name}_"):
                        team_jerseys.add(jersey)

                # Assign next available jersey number (1-99)
                for jersey_num in range(1, 100):
                    if jersey_num not in team_jerseys:
                        self._jersey_assignments[team_key] = jersey_num
                        break
                else:
                    # Fallback if all numbers are taken (shouldn't happen in practice)
                    self._jersey_assignments[team_key] = (obj_id % 99) + 1

            return self._jersey_assignments[team_key]

        except Exception as e:
            logging.warning(f"Jersey number assignment failed: {e}")
            return (obj_id % 99) + 1  # Fallback to simple modulo

    def _initialize_game_state(self):
        """Initialize game state tracking variables."""
        if not hasattr(self, '_game_state'):
            self._game_state = {
                'period': 1,
                'score_team_A': 0,
                'score_team_B': 0,
                'game_start_time': 0,
                'period_start_time': 0,
                'total_game_time': 0,
                'goals_scored': []
            }

    def _update_game_state(self, timestamp: float, events: List[Dict[str, Any]]):
        """Update game state based on detected events."""
        try:
            if not hasattr(self, '_game_state'):
                self._initialize_game_state()
                self._game_state['game_start_time'] = timestamp
                self._game_state['period_start_time'] = timestamp

            # Update total game time
            self._game_state['total_game_time'] = timestamp - self._game_state['game_start_time']

            # Check for goals in events
            for event in events:
                if event.get('event_type') == 'Shot' and event.get('event_outcome') == 'goal':
                    team = event.get('team', 'unknown')
                    if team in ['Team A', 'Team_A', 'Cluster 0']:
                        self._game_state['score_team_A'] += 1
                    elif team in ['Team B', 'Team_B', 'Cluster 1']:
                        self._game_state['score_team_B'] += 1

                    self._game_state['goals_scored'].append({
                        'timestamp': timestamp,
                        'team': team,
                        'player_id': event.get('object_id', -1)
                    })

            # Simple period management (45 minutes per half)
            if self._game_state['total_game_time'] > 45 * 60 and self._game_state['period'] == 1:
                self._game_state['period'] = 2
                self._game_state['period_start_time'] = timestamp
                logging.info("Switched to second half")

        except Exception as e:
            logging.warning(f"Game state update failed: {e}")

    def _determine_player_role(self, obj: Dict[str, Any], frame_data: Dict[str, Any]) -> str:
        """Determine player role based on position, team, and context."""
        try:
            pos_pitch = obj.get('pos_pitch', [0, 0])
            if not pos_pitch or len(pos_pitch) < 2:
                return 'unknown'

            x, y = pos_pitch
            team = obj.get('team', 'unknown')

            # Get pitch dimensions from homography manager
            pitch_width = self.homography_manager.pitch_dims[0] if hasattr(self, 'homography_manager') else 105
            pitch_height = self.homography_manager.pitch_dims[1] if hasattr(self, 'homography_manager') else 68

            # Determine which side of the pitch the team is defending
            # This is a simplified assumption - in reality, this should be determined from game context
            team_defending_left = team in ['Team A', 'Team_A', 'Cluster 0']  # Assumption

            if team_defending_left:
                # Team defends left side (x=0), attacks right side (x=pitch_width)
                if x < pitch_width * 0.25:  # Defensive third
                    if abs(y - pitch_height/2) < pitch_height * 0.2:  # Central
                        return 'CB'  # Center Back
                    else:
                        return 'FB'  # Full Back
                elif x > pitch_width * 0.75:  # Attacking third
                    if abs(y - pitch_height/2) < pitch_height * 0.3:  # Central
                        return 'ST'  # Striker
                    else:
                        return 'W'   # Winger
                else:  # Middle third
                    if abs(y - pitch_height/2) < pitch_height * 0.25:  # Central
                        return 'CM'  # Central Midfielder
                    else:
                        return 'WM'  # Wide Midfielder
            else:
                # Team defends right side (x=pitch_width), attacks left side (x=0)
                if x > pitch_width * 0.75:  # Defensive third
                    if abs(y - pitch_height/2) < pitch_height * 0.2:  # Central
                        return 'CB'  # Center Back
                    else:
                        return 'FB'  # Full Back
                elif x < pitch_width * 0.25:  # Attacking third
                    if abs(y - pitch_height/2) < pitch_height * 0.3:  # Central
                        return 'ST'  # Striker
                    else:
                        return 'W'   # Winger
                else:  # Middle third
                    if abs(y - pitch_height/2) < pitch_height * 0.25:  # Central
                        return 'CM'  # Central Midfielder
                    else:
                        return 'WM'  # Wide Midfielder

        except Exception as e:
            logging.warning(f"Player role determination failed: {e}")
            return 'unknown'

    def _calculate_player_speed(self, obj: Dict[str, Any], frame_data: Dict[str, Any]) -> str:
        """Calculate player speed in km/h using proper coordinate system and frame rate."""
        try:
            # Get tracking history for this object
            obj_id = obj.get('id', -1)
            if obj_id in self.object_tracker.tracking_history:
                history = self.object_tracker.tracking_history[obj_id]
                if len(history) >= 2:
                    # Get current and previous positions in pitch coordinates
                    current_pos = obj.get('pos_pitch', [0, 0])

                    # Find previous position in pitch coordinates from history
                    prev_pos = None
                    for i in range(len(history) - 2, -1, -1):  # Go backwards through history
                        prev_entry = history[i]
                        prev_bbox = prev_entry.get('bbox', [0, 0, 0, 0])

                        # Convert previous bbox to pitch coordinates using homography
                        if self.homography_manager.homography_matrix is not None:
                            try:
                                # Use bottom center of bbox for position
                                x_center = (prev_bbox[0] + prev_bbox[2]) / 2
                                y_bottom = prev_bbox[3]
                                video_point = np.array([[[x_center, y_bottom]]], dtype=np.float32)

                                pitch_point = cv2.perspectiveTransform(
                                    video_point, self.homography_manager.homography_matrix
                                )
                                prev_pos = pitch_point.flatten().tolist()
                                break
                            except Exception:
                                continue
                        else:
                            # Fallback: use simple scaling if no homography
                            video_width, video_height = 1920, 1080  # Default assumption
                            pitch_width, pitch_height = self.homography_manager.pitch_dims

                            x_center = (prev_bbox[0] + prev_bbox[2]) / 2
                            y_bottom = prev_bbox[3]

                            prev_pos = [
                                (x_center / video_width) * pitch_width,
                                (y_bottom / video_height) * pitch_height
                            ]
                            break

                    if current_pos and prev_pos and len(current_pos) >= 2 and len(prev_pos) >= 2:
                        # Calculate distance in meters (pitch coordinates should be in meters)
                        distance = ((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)**0.5

                        # Use actual video frame rate instead of hardcoded value
                        fps = self.fps if hasattr(self, 'fps') and self.fps > 0 else 30.0
                        time_diff = 1.0 / fps  # seconds between frames

                        # Calculate speed
                        speed_ms = distance / time_diff  # m/s
                        speed_kmh = speed_ms * 3.6  # km/h

                        # Apply reasonable limits (human running speed typically 0-40 km/h)
                        if speed_kmh > 50.0:  # Likely tracking error
                            return ""

                        return f"{speed_kmh:.1f}"

            return ""

        except Exception as e:
            logging.warning(f"Player speed calculation failed: {e}")
            return ""

    def _calculate_ball_speed(self, obj: Dict[str, Any], frame_data: Dict[str, Any]) -> str:
        """Calculate ball speed in km/h using proper coordinate system and frame rate."""
        try:
            # Get tracking history for this object
            obj_id = obj.get('id', -1)
            if obj_id in self.object_tracker.tracking_history:
                history = self.object_tracker.tracking_history[obj_id]
                if len(history) >= 2:
                    # Get current and previous positions in pitch coordinates
                    current_pos = obj.get('pos_pitch', [0, 0])

                    # Find previous position in pitch coordinates from history
                    prev_pos = None
                    for i in range(len(history) - 2, -1, -1):  # Go backwards through history
                        prev_entry = history[i]
                        prev_bbox = prev_entry.get('bbox', [0, 0, 0, 0])

                        # Convert previous bbox to pitch coordinates using homography
                        if self.homography_manager.homography_matrix is not None:
                            try:
                                # Use center of bbox for ball position
                                x_center = (prev_bbox[0] + prev_bbox[2]) / 2
                                y_center = (prev_bbox[1] + prev_bbox[3]) / 2
                                video_point = np.array([[[x_center, y_center]]], dtype=np.float32)

                                pitch_point = cv2.perspectiveTransform(
                                    video_point, self.homography_manager.homography_matrix
                                )
                                prev_pos = pitch_point.flatten().tolist()
                                break
                            except Exception:
                                continue
                        else:
                            # Fallback: use simple scaling if no homography
                            video_width, video_height = 1920, 1080  # Default assumption
                            pitch_width, pitch_height = self.homography_manager.pitch_dims

                            x_center = (prev_bbox[0] + prev_bbox[2]) / 2
                            y_center = (prev_bbox[1] + prev_bbox[3]) / 2

                            prev_pos = [
                                (x_center / video_width) * pitch_width,
                                (y_center / video_height) * pitch_height
                            ]
                            break

                    if current_pos and prev_pos and len(current_pos) >= 2 and len(prev_pos) >= 2:
                        # Calculate distance in meters
                        distance = ((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)**0.5

                        # Use actual video frame rate
                        fps = self.fps if hasattr(self, 'fps') and self.fps > 0 else 30.0
                        time_diff = 1.0 / fps  # seconds between frames

                        # Calculate speed
                        speed_ms = distance / time_diff  # m/s
                        speed_kmh = speed_ms * 3.6  # km/h

                        # Apply reasonable limits for ball speed (0-200 km/h for soccer)
                        if speed_kmh > 200.0:  # Likely tracking error
                            return ""

                        return f"{speed_kmh:.1f}"

            return ""

        except Exception as e:
            logging.warning(f"Ball speed calculation failed: {e}")
            return ""

    def _determine_events(self, obj: Dict[str, Any], actions: Dict[str, Any], all_objects: List[Dict[str, Any]]) -> Tuple[str, str, str]:
        """Determine event type, outcome, and pass recipient with improved logic."""
        try:
            event_type = ""
            event_outcome = ""
            pass_recipient_id = ""

            # Get pitch dimensions for goal detection
            pitch_width = self.homography_manager.pitch_dims[0] if hasattr(self, 'homography_manager') else 105
            pitch_height = self.homography_manager.pitch_dims[1] if hasattr(self, 'homography_manager') else 68

            # Goal areas (assuming goals are at x=0 and x=pitch_width)
            goal_area_depth = 5.5  # meters (standard goal area depth)
            goal_width = 7.32  # meters (standard goal width)
            goal_center_y = pitch_height / 2

            # Check if this is a ball object
            if obj.get('type') == 'sports_ball':
                ball_pos = obj.get('pos_pitch', [0, 0])
                if not ball_pos or len(ball_pos) < 2:
                    return event_type, event_outcome, pass_recipient_id

                # Find nearby players with proper distance calculation
                nearby_players = []
                for other_obj in all_objects:
                    if other_obj.get('type') == 'person':
                        other_pos = other_obj.get('pos_pitch', [0, 0])
                        if other_pos and len(other_pos) >= 2:
                            distance = ((ball_pos[0] - other_pos[0])**2 + (ball_pos[1] - other_pos[1])**2)**0.5
                            if distance < 3.0:  # Within 3 meters
                                nearby_players.append((other_obj, distance))

                # Sort by distance
                nearby_players.sort(key=lambda x: x[1])

                # Improved event logic
                if len(nearby_players) >= 2:
                    # Potential pass situation
                    player1, dist1 = nearby_players[0]
                    player2, dist2 = nearby_players[1]

                    # Check if players are from different teams (indicating a pass)
                    if player1.get('team') == player2.get('team') and dist1 < 2.0:
                        event_type = "Pass"
                        event_outcome = "successful"
                        pass_recipient_id = str(player2.get('id', ''))
                    elif dist1 < 1.5:  # Very close to one player
                        event_type = "Dribble"
                        event_outcome = "successful"

                elif len(nearby_players) == 1:
                    player, distance = nearby_players[0]
                    player_pos = player.get('pos_pitch', [0, 0])

                    if distance < 1.5:  # Player has control
                        # Check if near goal for shot detection
                        near_left_goal = (ball_pos[0] < goal_area_depth and
                                        abs(ball_pos[1] - goal_center_y) < goal_width/2 + 5)
                        near_right_goal = (ball_pos[0] > pitch_width - goal_area_depth and
                                         abs(ball_pos[1] - goal_center_y) < goal_width/2 + 5)

                        if near_left_goal or near_right_goal:
                            # Check ball speed for shot detection
                            speed_str = self._calculate_ball_speed(obj, {})
                            if speed_str:
                                try:
                                    speed = float(speed_str)
                                    if speed > 30:  # High speed indicates shot
                                        event_type = "Shot"
                                        # Improved goal detection
                                        in_left_goal = (ball_pos[0] <= 0 and
                                                      abs(ball_pos[1] - goal_center_y) < goal_width/2)
                                        in_right_goal = (ball_pos[0] >= pitch_width and
                                                       abs(ball_pos[1] - goal_center_y) < goal_width/2)
                                        event_outcome = "goal" if (in_left_goal or in_right_goal) else "miss"
                                except ValueError:
                                    pass

                        if not event_type:  # No shot detected
                            event_type = "Dribble"
                            event_outcome = "successful"

            elif obj.get('type') == 'person':
                # Player-centric events
                player_pos = obj.get('pos_pitch', [0, 0])
                if not player_pos or len(player_pos) < 2:
                    return event_type, event_outcome, pass_recipient_id

                # Find ball
                ball_obj = None
                ball_distance = float('inf')
                for other_obj in all_objects:
                    if other_obj.get('type') == 'sports_ball':
                        ball_pos = other_obj.get('pos_pitch', [0, 0])
                        if ball_pos and len(ball_pos) >= 2:
                            distance = ((player_pos[0] - ball_pos[0])**2 + (player_pos[1] - ball_pos[1])**2)**0.5
                            if distance < ball_distance:
                                ball_distance = distance
                                ball_obj = other_obj

                if ball_obj and ball_distance < 5.0:
                    if ball_distance < 1.5:  # Player has possession
                        # Determine action based on context and position
                        near_left_goal = (player_pos[0] < goal_area_depth * 2 and
                                        abs(player_pos[1] - goal_center_y) < goal_width + 10)
                        near_right_goal = (player_pos[0] > pitch_width - goal_area_depth * 2 and
                                         abs(player_pos[1] - goal_center_y) < goal_width + 10)

                        if near_left_goal or near_right_goal:
                            event_type = "Shot_Attempt"
                            event_outcome = "in_progress"
                        else:
                            event_type = "Possession"
                            event_outcome = "active"
                    elif ball_distance < 3.0:  # Player approaching ball
                        event_type = "Approach"
                        event_outcome = "in_progress"

            return event_type, event_outcome, pass_recipient_id

        except Exception as e:
            logging.warning(f"Event determination failed: {e}")
            return "", "", ""


if __name__ == '__main__':
    try:
        config = Config()
        # Load configuration from YAML
        config.load_from_yaml()
        
        # Validate configuration
        if not config.validate_config():
            exit(1)
        
        # Start processing
        processor = VideoProcessor(config)
        processor.start()
        
        logging.info("Processing completed successfully.")
        
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user.")
    except Exception as e:
        logging.critical(f"Critical error: {e}", exc_info=True)
        exit(1)
