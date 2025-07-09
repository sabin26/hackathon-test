from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np
import pandas as pd
import threading
import queue
import time
import os
import joblib
import hashlib
import logging
import json
import torch
from torchvision import transforms
from ultralytics import YOLO
from sklearn.cluster import KMeans
from collections import deque, defaultdict
import warnings
import psutil
import gc
import csv
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# Professional logging setup to provide clear, timestamped updates
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    handlers=[
        logging.FileHandler('sports_analytics.log'),
        logging.StreamHandler()
    ]
)

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class SystemMetrics:
    """Track system resource usage"""
    cpu_percent: float
    memory_percent: float
    available_memory_gb: float

# --- 1. ENTERPRISE CONFIGURATION ---
class Config:
    """Configuration with parameter hashing for robust checkpointing."""
    # --- MANDATORY: UPDATE THIS PATH ---
    VIDEO_PATH = 'sample_video.mp4'  # Change from placeholder path
    
    # --- OPTIONAL: Change output file name if desired ---
    OUTPUT_CSV_PATH = 'enterprise_analytics_output.csv'
    
    # --- Directory for storing learned models ---
    CHECKPOINT_DIR = './analyzer_checkpoints'
    
    # --- Key parameters that affect model state. Changing these will trigger re-calibration. ---
    MODEL_PARAMS = {
        'TEAM_N_CLUSTERS': 3,
        'TEAM_FEATURE_BINS': 16,
        'MIN_CROP_SIZE': 100,  # Minimum crop area in pixels
        'MAX_CROPS_PER_CLUSTER': 15,  # Increased for better HITL
        'FRAME_SKIP_INTERVAL': 2,  # Process every Nth frame for performance
        'MAX_PROCESSING_QUEUE_SIZE': 32,  # Reduced queue size
    }

    # --- Paths generated dynamically based on video and parameter hashes ---
    checkpoint_path_prefix: Optional[str] = None
    
    # --- Paths to AI models. Placeholders are used if files don't exist. ---
    YOLO_MODEL_PATH = 'yolov8n.pt'  # Updated to more recent model
    SEGMENTATION_MODEL_PATH = 'path/to/your/segmentation_model.pth'

    # --- System behavior parameters ---
    TEAM_SAMPLES_TO_COLLECT = 300  # Reduced for faster processing
    HOMOGRAPHY_RECAL_THRESHOLD = 3.0  # Max allowed reprojection error in pixels
    HOMOGRAPHY_CHECK_INTERVAL = 150  # Check quality every N frames
    ACTION_SEQUENCE_LENGTH = 30
    
    # --- New performance parameters ---
    FRAME_QUEUE_SIZE = 64  # Reduced to prevent memory issues
    PROCESSING_TIMEOUT = 1.0
    MAX_RETRIES = 3

    # --- Validation constants ---
    MIN_VIDEO_RESOLUTION = (320, 240)
    MAX_VIDEO_RESOLUTION = (4096, 2160)
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']

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
        # Check video file existence and format
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
        """A more realistic placeholder that finds lines on the green parts of the pitch."""
        try:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # More robust green detection with multiple ranges
            lower_green1 = np.array([35, 40, 40])
            upper_green1 = np.array([85, 255, 255])
            lower_green2 = np.array([45, 60, 60])
            upper_green2 = np.array([75, 255, 255])
            
            pitch_mask1 = cv2.inRange(hsv_frame, lower_green1, upper_green1)
            pitch_mask2 = cv2.inRange(hsv_frame, lower_green2, upper_green2)
            pitch_mask = cv2.bitwise_or(pitch_mask1, pitch_mask2)
            
            # Improved morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            pitch_mask = cv2.morphologyEx(pitch_mask, cv2.MORPH_CLOSE, kernel)
            pitch_mask = cv2.morphologyEx(pitch_mask, cv2.MORPH_OPEN, kernel)
            
            frame_on_pitch = cv2.bitwise_and(frame, frame, mask=pitch_mask)
            gray_on_pitch = cv2.cvtColor(frame_on_pitch, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur before edge detection
            gray_on_pitch = cv2.GaussianBlur(gray_on_pitch, (3, 3), 0)
            line_mask = cv2.Canny(gray_on_pitch, 50, 150, apertureSize=3)
            
            return line_mask
            
        except Exception as e:
            logging.error(f"Placeholder prediction failed: {e}")
            return np.zeros(frame.shape[:2], dtype=np.uint8)


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
                results = self.model.track(frame_resized, persist=True, 
                                         classes=[0, 32], verbose=False, 
                                         conf=0.3, iou=0.5)
                
                if results and results[0].boxes is not None:
                    # Ensure we have numpy arrays for iteration
                    def to_numpy(x):
                        if hasattr(x, 'cpu'):
                            return x.cpu().numpy()
                        return np.array(x)
                    
                    xyxy = to_numpy(results[0].boxes.xyxy)
                    ids = to_numpy(results[0].boxes.id)
                    clss = to_numpy(results[0].boxes.cls)
                    confs = to_numpy(results[0].boxes.conf)
                    
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

    def calculate_homography(self, frame: np.ndarray) -> bool:
        """Calculates homography using the segmentation mask with improved robustness."""
        if self.calibration_attempts >= self.max_calibration_attempts:
            logging.warning("Max calibration attempts reached")
            return False
            
        self.calibration_attempts += 1
        
        try:
            line_mask = self.segmentation_model.predict(frame)
            if line_mask is None or line_mask.size == 0:
                return False
                
            # Improved line detection with multiple parameter sets
            lines = self._detect_lines_robust(line_mask)
            if lines is None or len(lines) < 4:
                return False

            # Separate horizontal and vertical lines with better angle thresholds
            h_lines, v_lines = self._separate_lines(lines)
            
            if len(h_lines) < 2 or len(v_lines) < 2:
                return False

            # Find pitch boundaries more robustly
            pitch_corners = self._find_pitch_corners(h_lines, v_lines)
            if pitch_corners is None:
                return False

            self.last_src_points = np.array(pitch_corners, dtype=np.float32)
            self.homography_matrix, _ = cv2.findHomography(
                self.last_src_points, 
                self.template_points, 
                cv2.RANSAC, 
                5.0
            )
            
            if self.homography_matrix is not None:
                logging.info("Homography calibrated successfully.")
                self.calibration_attempts = 0  # Reset on success
                return True
                
        except Exception as e:
            logging.error(f"Homography calculation failed: {e}")
            
        return False

    def _detect_lines_robust(self, line_mask: np.ndarray) -> Optional[np.ndarray]:
        """Detect lines with multiple parameter sets for robustness."""
        param_sets = [
            (50, 50, 10),  # threshold, minLineLength, maxLineGap
            (30, 30, 15),
            (70, 70, 5)
        ]
        
        for threshold, min_length, max_gap in param_sets:
            lines = cv2.HoughLinesP(
                line_mask, 1, np.pi / 180, 
                threshold=threshold, 
                minLineLength=min_length, 
                maxLineGap=max_gap
            )
            if lines is not None and len(lines) >= 4:
                return lines
                
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
            return tracked_objects
            
        try:
            for obj in tracked_objects:
                x1, y1, x2, y2 = obj['bbox_video']
                
                # Use bottom center of bounding box
                bottom_center = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
                
                transformed_point = cv2.perspectiveTransform(
                    bottom_center.reshape(-1, 1, 2), 
                    self.homography_matrix
                )
                
                obj['pos_pitch'] = transformed_point.flatten().tolist()
                
        except Exception as e:
            logging.error(f"Homography application failed: {e}")
            
        return tracked_objects


# --- 5. TEAM IDENTIFIER ---
class TeamIdentifier:
    """Efficiently collects crops for a superior HITL experience."""
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
        """Collects feature vector and corresponding crop image with thread safety."""
        with self._lock:
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
    def __init__(self, config: Config):
        self.config = config
        self.cap = None
        self.video_hash = None
        self.frame_queue = queue.Queue(maxsize=config.MODEL_PARAMS['MAX_PROCESSING_QUEUE_SIZE'])
        self.results_data: List[Dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.processing_lock = threading.Lock()
        self.frame_skip_counter = 0
        
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

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logging.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop_event.set()

    def _initialize_video(self) -> None:
        """Initialize video capture with error handling."""
        try:
            self.cap = cv2.VideoCapture(self.config.VIDEO_PATH)
            if not self.cap.isOpened():
                raise IOError(f"Cannot open video file: {self.config.VIDEO_PATH}")
                
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logging.info(f"Video: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
            
            # Calculate robust video hash
            self.video_hash = self._calculate_robust_video_hash()
            self.config.generate_checkpoint_prefix(self.video_hash)
            
        except Exception as e:
            logging.error(f"Video initialization failed: {e}")
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
            
            logging.info("All components initialized successfully")
            
        except Exception as e:
            logging.error(f"Component initialization failed: {e}")
            raise

    def _calculate_robust_video_hash(self) -> str:
        """Hashes frames from start, middle, and end for maximum robustness."""
        logging.info("Calculating robust video hash...")
        
        try:
            hasher = hashlib.md5()
            
            # Handle edge cases for very short videos
            if self.total_frames <= 0:
                # Fallback: use video file stats
                stat = os.stat(self.config.VIDEO_PATH)
                hasher.update(str(stat.st_size).encode())
                hasher.update(str(stat.st_mtime).encode())
                return hasher.hexdigest()[:16]
            
            # Sample frames intelligently
            sample_indices = [0]
            if self.total_frames > 2:
                sample_indices.append(self.total_frames // 2)
            if self.total_frames > 1:
                sample_indices.append(self.total_frames - 1)
            
            for idx in sample_indices:
                if self.cap is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = self.cap.read()
                    if ret:
                        # Hash a downsampled version for consistency
                        small_frame = cv2.resize(frame, (64, 64))
                        hasher.update(small_frame.tobytes())
                
            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind for processing
            return hasher.hexdigest()[:16]
            
        except Exception as e:
            logging.error(f"Video hash calculation failed: {e}")
            return hashlib.md5(str(time.time()).encode()).hexdigest()[:16]

    def load_state(self) -> None:
        """Loads learned models from configuration-aware checkpoints."""
        try:
            team_model_path = f"{self.config.checkpoint_path_prefix}_team_model.joblib"
            homography_path = f"{self.config.checkpoint_path_prefix}_homography.npy"
            
            if os.path.exists(team_model_path):
                self.team_identifier = joblib.load(team_model_path)
                logging.info("Loaded team identification model from checkpoint.")
                
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
                joblib.dump(self.team_identifier, team_model_path)
                logging.info(f"Saved team model to {team_model_path}")
                
            if self.homography_manager.homography_matrix is not None:
                np.save(homography_path, self.homography_manager.homography_matrix)
                logging.info(f"Saved homography matrix to {homography_path}")
                
        except Exception as e:
            logging.error(f"State saving failed: {e}")

    def _perform_visual_hitl(self) -> None:
        """A superior HITL experience with a visual montage."""
        try:
            logging.warning("PAUSING FOR HUMAN INPUT: Please label the teams in the popup window.")
            
            montages = []
            cluster_ids = sorted(self.team_identifier.example_crops.keys())
            
            for cluster_id in cluster_ids:
                crops = self.team_identifier.example_crops[cluster_id]
                if not crops:
                    continue
                
                # Create a grid of crops instead of a single row
                grid_size = min(len(crops), 10)
                rows = []
                
                for i in range(0, grid_size, 5):
                    row_crops = crops[i:i+5]
                    if len(row_crops) > 0:
                        row = cv2.hconcat(row_crops)
                        rows.append(row)
                
                if rows:
                    crop_grid = cv2.vconcat(rows)
                    
                    # Add label
                    label_img = np.zeros((40, crop_grid.shape[1], 3), dtype=np.uint8)
                    cv2.putText(label_img, f"Cluster {cluster_id}", (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    montages.append(cv2.vconcat([label_img, crop_grid]))
            
            if not montages:
                logging.error("Could not generate HITL montage: no example crops were collected.")
                return

            final_montage = cv2.vconcat(montages)
            
            # Resize if too large
            if final_montage.shape[0] > 800:
                scale = 800 / final_montage.shape[0]
                new_width = int(final_montage.shape[1] * scale)
                final_montage = cv2.resize(final_montage, (new_width, 800))
            
            cv2.imshow("Human-in-the-Loop: Label Teams", final_montage)
            cv2.waitKey(100)  # Brief pause to ensure window is displayed
            
            print("\n" + "="*60)
            print("ACTION REQUIRED: A window named 'Human-in-the-Loop' has opened.")
            print("Based on the image, please provide labels for the clusters.")
            print("Press any key in the image window after reading, then provide labels.")
            
            cv2.waitKey(0)  # Wait for key press
            
            user_input = input(f"Enter {self.config.MODEL_PARAMS['TEAM_N_CLUSTERS']} comma-separated labels (e.g., Team A,Team B,Referee): ")
            print("="*60 + "\n")

            cv2.destroyWindow("Human-in-the-Loop: Label Teams")
            
            labels = [label.strip() for label in user_input.split(',')]
            if len(labels) == self.config.MODEL_PARAMS['TEAM_N_CLUSTERS']:
                self.team_identifier.team_map = {cluster_ids[i]: labels[i] for i in range(len(labels))}
                logging.info(f"Team labels received and applied: {self.team_identifier.team_map}")
            else:
                logging.error(f"Incorrect number of labels ({len(labels)} vs {self.config.MODEL_PARAMS['TEAM_N_CLUSTERS']}). Labeling aborted.")
                
        except Exception as e:
            logging.error(f"HITL process failed: {e}")

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
                            
                            # Perform HITL if needed
                            if not self.team_identifier.team_map:
                                self._perform_visual_hitl()
                        except Exception as e:
                            logging.error(f"Team model fitting failed: {e}")

                # Homography management with better error recovery
                if self.homography_manager.homography_matrix is None:
                    try:
                        self.homography_manager.calculate_homography(frame)
                    except Exception as e:
                        logging.error(f"Homography calculation failed: {e}")
                
                # Periodically check homography quality
                if actual_frame_id > 0 and actual_frame_id % self.config.HOMOGRAPHY_CHECK_INTERVAL == 0:
                    try:
                        error = self.homography_manager.calculate_reprojection_error()
                        if error > self.config.HOMOGRAPHY_RECAL_THRESHOLD:
                            logging.warning(f"High reprojection error ({error:.2f}px). Triggering recalibration.")
                            self.homography_manager.homography_matrix = None
                    except Exception as e:
                        logging.warning(f"Homography error check failed: {e}")

                # Classify team members
                for obj in objects:
                    if obj['type'] == 'person':
                        try:
                            obj['team'] = self.team_identifier.classify_player(frame, obj['bbox_video'])
                        except Exception as e:
                            logging.warning(f"Team classification failed: {e}")
                            obj['team'] = "Unknown"
                
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
                with self.processing_lock:
                    self.results_data.append({
                        "frame_id": actual_frame_id,
                        "timestamp": actual_frame_id / self.fps if self.fps > 0 else 0,
                        "actions": actions,
                        "objects": objects
                    })
                
                # Track performance metrics
                processing_time = time.time() - start_time
                self.performance_metrics['processing_times'].append(processing_time)
                self.performance_metrics['frames_processed'] += 1
                
                if processing_time > 1.0:  # Log slow frames
                    logging.warning(f"Slow processing: {processing_time:.2f}s for frame {actual_frame_id}")
                
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
                        ret, frame = self.cap.read()
                        if not ret:
                            logging.info("End of video file reached.")
                            self.stop_event.set()
                            time.sleep(1)  # Give processing thread time to finish
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

                # Check for quit signal
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("'q' pressed, shutting down.")
                    self.stop_event.set()
                    break
            
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
            
            # Export data
            self._export_data()
            
        except Exception as e:
            logging.error(f"Cleanup failed: {e}")

    def _export_data(self) -> None:
        """Fixed and enhanced data export with complete implementation"""
        if not self.results_data:
            logging.warning("No data was processed to export.")
            return
        
        try:
            logging.info(f"Exporting {len(self.results_data)} frames of data...")
            
            # Create backup of existing file
            if os.path.exists(self.config.OUTPUT_CSV_PATH):
                backup_path = f"{self.config.OUTPUT_CSV_PATH}.backup_{int(time.time())}"
                os.rename(self.config.OUTPUT_CSV_PATH, backup_path)
                logging.info(f"Created backup: {backup_path}")
            
            export_data = []
            
            # Add data validation and complete the export implementation
            valid_frames = 0
            for frame_data in self.results_data:
                if not isinstance(frame_data, dict):
                    logging.warning(f"Invalid frame data type: {type(frame_data)}")
                    continue
                
                frame_id = frame_data.get('frame_id', -1)
                if frame_id < 0:
                    logging.warning(f"Invalid frame_id: {frame_id}")
                    continue
                
                # Extract object data
                objects = frame_data.get('objects', [])
                actions = frame_data.get('actions', {})
                timestamp = frame_data.get('timestamp', 0)
                
                # Create a row for each object
                if objects:
                    for obj in objects:
                        try:
                            row = {
                                'frame_id': frame_id,
                                'timestamp': timestamp,
                                'object_id': obj.get('id', -1),
                                'object_type': obj.get('type', 'unknown'),
                                'team': obj.get('team', 'unknown'),
                                'confidence': obj.get('confidence', 0.0),
                                'bbox_x1': obj.get('bbox_video', [0, 0, 0, 0])[0],
                                'bbox_y1': obj.get('bbox_video', [0, 0, 0, 0])[1],
                                'bbox_x2': obj.get('bbox_video', [0, 0, 0, 0])[2],
                                'bbox_y2': obj.get('bbox_video', [0, 0, 0, 0])[3],
                                'pitch_x': obj.get('pos_pitch', [0, 0])[0] if obj.get('pos_pitch') else None,
                                'pitch_y': obj.get('pos_pitch', [0, 0])[1] if obj.get('pos_pitch') else None,
                                'tracking_quality': obj.get('tracking_quality', 0.0),
                                'action_event': actions.get('event', ''),
                                'action_velocity': actions.get('velocity', 0.0)
                            }
                            export_data.append(row)
                        except Exception as e:
                            logging.warning(f"Failed to process object in frame {frame_id}: {e}")
                else:
                    # Create a row even if no objects detected
                    row = {
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'object_id': -1,
                        'object_type': 'none',
                        'team': 'none',
                        'confidence': 0.0,
                        'bbox_x1': 0, 'bbox_y1': 0, 'bbox_x2': 0, 'bbox_y2': 0,
                        'pitch_x': None, 'pitch_y': None,
                        'tracking_quality': 0.0,
                        'action_event': actions.get('event', ''),
                        'action_velocity': actions.get('velocity', 0.0)
                    }
                    export_data.append(row)
                
                valid_frames += 1
            
            if valid_frames == 0:
                logging.error("No valid frames to export")
                return
            
            # Create DataFrame and validate
            df = pd.DataFrame(export_data)
            
            # Check for required columns
            required_columns = ['frame_id', 'timestamp', 'object_type']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logging.error(f"Missing required columns: {missing_columns}")
                return
            
            # Add metadata to CSV
            metadata_rows = [
                ['# Video Processing Metadata'],
                [f'# Video Path: {self.config.VIDEO_PATH}'],
                [f'# Processing Date: {time.strftime("%Y-%m-%d %H:%M:%S")}'],
                [f'# Total Frames: {self.total_frames}'],
                [f'# Processed Frames: {valid_frames}'],
                [f'# Video Hash: {self.video_hash}'],
                [f'# Configuration: {json.dumps(self.config.MODEL_PARAMS)}'],
                ['# End Metadata', ''],
            ]
            
            # Write metadata and data
            with open(self.config.OUTPUT_CSV_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                for row in metadata_rows:
                    writer.writerow(row)
                
            # Append DataFrame
            df.to_csv(self.config.OUTPUT_CSV_PATH, mode='a', index=False)
            
            logging.info(f"Successfully exported {len(export_data)} rows to {self.config.OUTPUT_CSV_PATH}")
            
            # Verify export
            try:
                verification_df = pd.read_csv(self.config.OUTPUT_CSV_PATH, comment='#')
                if len(verification_df) != len(export_data):
                    logging.warning("Export verification failed: row count mismatch")
                else:
                    logging.info("Export verification successful")
            except Exception as e:
                logging.warning(f"Export verification failed: {e}")
            
        except Exception as e:
            logging.error(f"Data export failed: {e}")
            # Try to restore backup if export failed
            backup_files = [f for f in os.listdir('.') 
                           if f.startswith(os.path.basename(self.config.OUTPUT_CSV_PATH) + '.backup_')]
            if backup_files:
                latest_backup = max(backup_files)
                try:
                    os.rename(latest_backup, self.config.OUTPUT_CSV_PATH)
                    logging.info(f"Restored backup: {latest_backup}")
                except Exception as restore_error:
                    logging.error(f"Failed to restore backup: {restore_error}")


if __name__ == '__main__':
    try:
        config = Config()
        
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
