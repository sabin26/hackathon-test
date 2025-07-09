from typing import Optional
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

# Professional logging setup to provide clear, timestamped updates
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'
)

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
    }

    # --- Paths generated dynamically based on video and parameter hashes ---
    checkpoint_path_prefix: Optional[str] = None
    
    # --- Paths to AI models. Placeholders are used if files don't exist. ---
    YOLO_MODEL_PATH = 'yolov13n.pt'  # Use a fine-tuned model in production
    SEGMENTATION_MODEL_PATH = 'path/to/your/segmentation_model.pth'

    # --- System behavior parameters ---
    TEAM_SAMPLES_TO_COLLECT = 500
    HOMOGRAPHY_RECAL_THRESHOLD = 3.0  # Max allowed reprojection error in pixels
    HOMOGRAPHY_CHECK_INTERVAL = 150  # Check quality every N frames
    ACTION_SEQUENCE_LENGTH = 30

    @classmethod
    def generate_checkpoint_prefix(cls, video_hash: str):
        """Creates a prefix based on video and parameter hashes for robust state management."""
        # Create a stable string representation of the parameters dictionary
        params_str = json.dumps(cls.MODEL_PARAMS, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        cls.checkpoint_path_prefix = os.path.join(cls.CHECKPOINT_DIR, f'{video_hash}_{params_hash}')
        logging.info(f"Using checkpoint prefix for this configuration: {cls.checkpoint_path_prefix}")


# --- 2. DEEP LEARNING MODEL BLUEPRINTS ---
class SegmentationModel:
    """Blueprint for a real DL segmentation model, with an advanced placeholder."""
    def __init__(self, model_path):
        self.model = None
        self.device = None
        self.transform = None
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            logging.info(f"Successfully loaded segmentation model '{model_path}' on {self.device}")
            # Standard image transformations for vision models
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 256), antialias=True),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        except FileNotFoundError:
            self.model = None
            logging.warning(f"Segmentation model not found at '{model_path}'.")
            logging.info("Switching to ADVANCED PLACEHOLDER mode (green pitch detection).")
        except Exception as e:
            self.model = None
            logging.error(f"Failed to load segmentation model due to an error: {e}", exc_info=True)


    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Runs inference with the DL model or falls back to the placeholder."""
        if self.model and self.transform:
            # Real DL Inference Workflow
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(rgb_frame)
            if isinstance(input_tensor, np.ndarray):
                input_tensor = torch.from_numpy(input_tensor)
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
            
            mask = torch.argmax(output, dim=0).detach().cpu().numpy().astype(np.uint8)
            # Resize mask back to original frame size
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Assume class '1' corresponds to pitch lines in the trained model
            line_mask = np.where(mask_resized == 1, 255, 0).astype(np.uint8)
            return line_mask
        else:
            # Advanced Placeholder: Detects green pitch area and finds lines within it
            return self._placeholder_predict(frame)

    def _placeholder_predict(self, frame: np.ndarray) -> np.ndarray:
        """A more realistic placeholder that finds lines on the green parts of the pitch."""
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        pitch_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        
        kernel = np.ones((5, 5), np.uint8)
        pitch_mask = cv2.erode(pitch_mask, kernel, iterations=1)
        pitch_mask = cv2.dilate(pitch_mask, kernel, iterations=2)
        
        frame_on_pitch = cv2.bitwise_and(frame, frame, mask=pitch_mask)
        gray_on_pitch = cv2.cvtColor(frame_on_pitch, cv2.COLOR_BGR2GRAY)
        
        line_mask = cv2.Canny(gray_on_pitch, 60, 180)
        return line_mask


# --- 3. OBJECT TRACKER ---
class ObjectTracker:
    """Encapsulates the YOLO object tracker."""
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        logging.info(f"Object tracker initialized with model: {model_path}")

    def track_objects(self, frame: np.ndarray) -> list:
        """Detects and tracks objects, returning structured data."""
        results = self.model.track(frame, persist=True, classes=[0, 32], verbose=False)
        objects = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Ensure we have numpy arrays for iteration
            def to_numpy(x):
                if hasattr(x, 'cpu'):
                    return x.cpu().numpy()
                return np.array(x)
            xyxy = to_numpy(results[0].boxes.xyxy)
            ids = to_numpy(results[0].boxes.id)
            clss = to_numpy(results[0].boxes.cls)
            for box, track_id, cls_id in zip(xyxy, ids, clss):
                objects.append({
                    "id": int(track_id),
                    "type": self.model.names[int(cls_id)],
                    "bbox_video": [int(coord) for coord in box]
                })
        return objects


# --- 4. HOMOGRAPHY MANAGER ---
class HomographyManager:
    """Manages homography with correct reprojection error calculation."""
    def __init__(self, pitch_dims, segmentation_model):
        self.pitch_dims = pitch_dims
        self.segmentation_model = segmentation_model
        self.homography_matrix = None
        self.template_points = np.array([
            [0, 0], [pitch_dims[0], 0], 
            [pitch_dims[0], pitch_dims[1]], [0, pitch_dims[1]]
        ], dtype=np.float32)
        self.last_src_points = None

    def calculate_homography(self, frame: np.ndarray) -> bool:
        """Calculates homography using the segmentation mask."""
        line_mask = self.segmentation_model.predict(frame)
        lines = cv2.HoughLinesP(line_mask, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        if lines is None: 
            return False

        h_lines, v_lines = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 15: 
                h_lines.append(line)
            elif abs(abs(angle) - 90) < 15: 
                v_lines.append(line)

        if not h_lines or not v_lines: 
            return False

        top_line = min(h_lines, key=lambda line:
                       line[0][1])[0]
        bottom_line = max(h_lines, key=lambda line:
                          line[0][3])[0]
        left_line = min(v_lines, key=lambda line:
                         line[0][0])[0]
        right_line = max(v_lines, key=lambda line:
                          line[0][2])[0]

        def line_intersection(l1, l2):
            x1, y1, x2, y2 = l1
            x3, y3, x4, y4 = l2
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if den == 0:
                return None
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            px, py = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
            return [px, py]

        tl = line_intersection(top_line, left_line)
        tr = line_intersection(top_line, right_line)
        bl = line_intersection(bottom_line, left_line)
        br = line_intersection(bottom_line, right_line)

        # Ensure all points are not None before converting to numpy array
        src_points = [tl, tr, br, bl]
        if not all(src_points): 
            return False
        self.last_src_points = np.array(src_points, dtype=np.float32)
        self.homography_matrix, _ = cv2.findHomography(self.last_src_points, self.template_points, cv2.RANSAC, 5.0)
        logging.info("Homography calibrated successfully.")
        return True

    def calculate_reprojection_error(self) -> float:
        """Calculates the true reprojection error to monitor homography quality."""
        if self.homography_matrix is None or self.last_src_points is None:
            return float('inf')
        
        try:
            inv_homography = np.linalg.inv(self.homography_matrix)
            reprojected_points = cv2.perspectiveTransform(self.template_points.reshape(-1, 1, 2), inv_homography)
            error = np.mean(np.linalg.norm(self.last_src_points - reprojected_points.reshape(4, 2), axis=1))
            return error
        except np.linalg.LinAlgError:
            logging.warning("Could not invert homography matrix for error check.")
            return float('inf')
            
    def apply_homography(self, tracked_objects: list) -> list:
        """Transforms object coordinates from video space to pitch space."""
        if self.homography_matrix is None:
            return tracked_objects
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox_video']
            bottom_center = np.array([[(x1 + x2) / 2, y2]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(bottom_center.reshape(-1, 1, 2), self.homography_matrix)
            obj['pos_pitch'] = transformed_point.flatten().tolist()
        return tracked_objects


# --- 5. TEAM IDENTIFIER ---
class TeamIdentifier:
    """Efficiently collects crops for a superior HITL experience."""
    def __init__(self, n_clusters):
        self.kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        self.team_map: Optional[dict] = None
        self.is_fitted = False
        self.feature_samples = []
        self.example_crops = defaultdict(list)
        self.crop_samples = []  # Store crops alongside features

    def _extract_player_features(self, frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
        """Creates a feature vector (color histogram) from a player's torso."""
        x1, y1, x2, y2 = bbox
        h, w = y2 - y1, x2 - x1
        crop_bgr = frame[y1 + int(h * 0.2):y1 + int(h * 0.6), x1 + int(w * 0.2):x1 + int(w * 0.8)]
        if crop_bgr.size < 10:
            return None
        
        crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        bins = [Config.MODEL_PARAMS['TEAM_FEATURE_BINS']] * 2
        hist = cv2.calcHist([crop_hsv], [0, 1], None, bins, [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def collect_and_store_sample(self, frame: np.ndarray, bbox: list):
        """Collects feature vector and corresponding crop image."""
        features = self._extract_player_features(frame, bbox)
        if features is not None:
            self.feature_samples.append(features)
            
            # Always collect crop for HITL display
            x1, y1, x2, y2 = bbox
            h, w = y2 - y1, x2 - x1
            crop_bgr = frame[y1 + int(h * 0.2):y1 + int(h * 0.6), x1 + int(w * 0.2):x1 + int(w * 0.8)]
            if crop_bgr.size > 0:
                self.crop_samples.append(cv2.resize(crop_bgr, (50, 50)))
            else:
                self.crop_samples.append(np.zeros((50, 50, 3), dtype=np.uint8))
            
            if self.is_fitted:
                cluster_id = self.kmeans.predict([features])[0]
                if len(self.example_crops[cluster_id]) < 10:
                    self.example_crops[cluster_id].append(cv2.resize(crop_bgr, (50, 50)))

    def fit(self):
        """Fits the K-Means model and organizes crops by cluster."""
        logging.info("Fitting team feature model...")
        self.kmeans.fit(self.feature_samples)
        self.is_fitted = True
        
        # Organize existing crops by cluster
        for i, features in enumerate(self.feature_samples):
            if i < len(self.crop_samples):
                cluster_id = self.kmeans.predict([features])[0]
                if len(self.example_crops[cluster_id]) < 10:
                    self.example_crops[cluster_id].append(self.crop_samples[i])
        logging.info("Team K-Means model fitted. Ready for HITL or loading labels.")

    def classify_player(self, frame: np.ndarray, bbox: list) -> str:
        """Classifies a single player based on the fitted model and labeled map."""
        if not self.is_fitted:
            return "Unknown"
        
        features = self._extract_player_features(frame, bbox)
        if features is None:
            return "Unknown"
        
        pred_cluster = self.kmeans.predict([features])[0]
        return self.team_map.get(pred_cluster, f"Cluster {pred_cluster}") if self.team_map else f"Cluster {pred_cluster}"


# --- 6. ACTION RECOGNIZER HARNESS ---
class ActionRecognizer:
    """Harness for a sequence-based action recognition model."""
    def __init__(self):
        self.sequence_model = None # Placeholder for a loaded Keras/PyTorch LSTM/Transformer
        self.feature_sequence = deque(maxlen=Config.ACTION_SEQUENCE_LENGTH)
        logging.info("ActionRecognizer harness initialized. Needs a trained sequence model to be effective.")

    def recognize(self, tracked_objects: list) -> dict:
        """Collects sequences and would run inference if a model were loaded."""
        # This is a simplified placeholder. A real implementation would have richer features.
        ball = next((obj for obj in tracked_objects if obj.get('type') == 'sports ball' and obj.get('pos_pitch')), None)
        
        if ball:
            self.feature_sequence.append(ball['pos_pitch'])
        else: # If ball is not visible, append a placeholder
            self.feature_sequence.append(None)
        
        # Once the deque is full, you would run the model
        if len(self.feature_sequence) == Config.ACTION_SEQUENCE_LENGTH:
            if self.sequence_model:
                # Real inference code would go here
                pass
            else:
                # Simple heuristic as a fallback
                valid_positions = [p for p in self.feature_sequence if p is not None]
                if len(valid_positions) > 10:
                    velocity = np.linalg.norm(np.array(valid_positions[-1]) - np.array(valid_positions[0]))
                    if velocity > 500: # Arbitrary threshold
                        return {"event": "High-Speed Ball Movement"}
        return {}


# --- 7. MAIN VIDEO PROCESSOR ORCHESTRATOR ---
class VideoProcessor:
    """The main orchestrator for the entire analytics pipeline."""
    def __init__(self, config: Config):
        self.config = config
        self.cap = cv2.VideoCapture(config.VIDEO_PATH)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {config.VIDEO_PATH}")

        self.video_hash = self._calculate_robust_video_hash()
        self.config.generate_checkpoint_prefix(self.video_hash)
        
        self.frame_queue = queue.Queue(maxsize=128)
        self.results_data = []
        self.stop_event = threading.Event()

        # Initialize all components
        self.segmentation_model = SegmentationModel(config.SEGMENTATION_MODEL_PATH)
        self.homography_manager = HomographyManager(
            pitch_dims=(1050, 680),
            segmentation_model=self.segmentation_model
        )
        self.team_identifier = TeamIdentifier(config.MODEL_PARAMS['TEAM_N_CLUSTERS'])
        self.object_tracker = ObjectTracker(config.YOLO_MODEL_PATH)
        self.action_recognizer = ActionRecognizer()
        
        self.load_state()

    def _calculate_robust_video_hash(self) -> str:
        """Hashes frames from start, middle, and end for maximum robustness."""
        logging.info("Calculating robust video hash...")
        hasher = hashlib.md5()
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Handle edge cases for very short videos
        if total_frames <= 0:
            # Fallback: use video file stats
            stat = os.stat(self.config.VIDEO_PATH)
            hasher.update(str(stat.st_size).encode())
            hasher.update(str(stat.st_mtime).encode())
            return hasher.hexdigest()[:16]
        
        # Sample frames intelligently
        sample_indices = [0]
        if total_frames > 2:
            sample_indices.append(total_frames // 2)
        if total_frames > 1:
            sample_indices.append(total_frames - 1)
        
        for idx in sample_indices:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                # Hash a downsampled version for consistency
                small_frame = cv2.resize(frame, (64, 64))
                hasher.update(small_frame.tobytes())
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind for processing
        return hasher.hexdigest()[:16]

    def load_state(self):
        """Loads learned models from configuration-aware checkpoints."""
        team_model_path = f"{self.config.checkpoint_path_prefix}_team_model.joblib"
        homography_path = f"{self.config.checkpoint_path_prefix}_homography.npy"
        
        if os.path.exists(team_model_path):
            self.team_identifier = joblib.load(team_model_path)
            logging.info("Loaded team identification model from checkpoint.")
        if os.path.exists(homography_path):
            self.homography_manager.homography_matrix = np.load(homography_path)
            logging.info("Loaded homography matrix from checkpoint.")

    def save_state(self):
        """Saves learned models to disk for future runs."""
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)
        team_model_path = f"{self.config.checkpoint_path_prefix}_team_model.joblib"
        homography_path = f"{self.config.checkpoint_path_prefix}_homography.npy"
        
        if self.team_identifier.is_fitted and self.team_identifier.team_map:
            joblib.dump(self.team_identifier, team_model_path)
            logging.info(f"Saved team model to {team_model_path}")
        if self.homography_manager.homography_matrix is not None:
            np.save(homography_path, self.homography_manager.homography_matrix)
            logging.info(f"Saved homography matrix to {homography_path}")

    def _perform_visual_hitl(self):
        """A superior HITL experience with a visual montage."""
        logging.warning("PAUSING FOR HUMAN INPUT: Please label the teams in the popup window.")
        
        montages = []
        cluster_ids = sorted(self.team_identifier.example_crops.keys())
        for cluster_id in cluster_ids:
            crops = self.team_identifier.example_crops[cluster_id]
            if not crops:
                continue
            
            row = cv2.hconcat(crops)
            label_img = np.zeros((30, row.shape[1], 3), dtype=np.uint8)
            cv2.putText(label_img, f"Cluster {cluster_id}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            montages.append(cv2.vconcat([label_img, row]))
        
        if not montages:
            logging.error("Could not generate HITL montage: no example crops were collected.")
            return

        final_montage = cv2.vconcat(montages)
        cv2.imshow("Human-in-the-Loop: Label Teams", final_montage)
        
        print("\n" + "="*60)
        print("ACTION REQUIRED: A window named 'Human-in-the-Loop' has opened.")
        print("Based on the image, please provide labels for the clusters.")
        user_input = input(f"Enter {self.config.MODEL_PARAMS['TEAM_N_CLUSTERS']} comma-separated labels (e.g., Team A,Team B,Referee): ")
        print("="*60 + "\n")

        cv2.destroyWindow("Human-in-the-Loop: Label Teams")
        
        labels = [label.strip() for label in user_input.split(',')]
        if len(labels) == self.config.MODEL_PARAMS['TEAM_N_CLUSTERS']:
            self.team_identifier.team_map = {cluster_ids[i]: labels[i] for i in range(len(labels))}
            logging.info(f"Team labels received and applied: {self.team_identifier.team_map}")
        else:
            logging.error("Incorrect number of labels. Labeling aborted.")

    def _processing_loop(self):
        """The core processing logic with the corrected HITL preparation."""
        frame_id = 0
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue

            objects = self.object_tracker.track_objects(frame)

            # --- Corrected Team ID and HITL workflow ---
            if not self.team_identifier.is_fitted:
                # Phase 1: Collect samples
                if len(self.team_identifier.feature_samples) < self.config.TEAM_SAMPLES_TO_COLLECT:
                    for obj in objects:
                        if obj['type'] == 'person':
                            self.team_identifier.collect_and_store_sample(frame, obj['bbox_video'])
                
                # Phase 2: Fit model and prepare for HITL when enough samples are collected
                else:
                    self.team_identifier.fit()
                    
                    # Phase 3: Trigger HITL if no labels were loaded from a checkpoint
                    if not self.team_identifier.team_map:
                        self._perform_visual_hitl()

            # ... (Rest of the processing loop is identical: homography, classification, etc.) ...
            if self.homography_manager.homography_matrix is None:
                self.homography_manager.calculate_homography(frame)
            
            if frame_id > 0 and frame_id % self.config.HOMOGRAPHY_CHECK_INTERVAL == 0:
                error = self.homography_manager.calculate_reprojection_error()
                if error > self.config.HOMOGRAPHY_RECAL_THRESHOLD:
                    logging.warning(f"High reprojection error ({error:.2f}px). Triggering recalibration.")
                    self.homography_manager.homography_matrix = None

            for obj in objects:
                if obj['type'] == 'person':
                    obj['team'] = self.team_identifier.classify_player(frame, obj['bbox_video'])
            
            objects = self.homography_manager.apply_homography(objects)
            actions = self.action_recognizer.recognize(objects)

            self.results_data.append({"frame_id": frame_id, "actions": actions, "objects": objects})
            frame_id += 1

    def start(self):
        """Starts the video reading and processing threads."""
        processing_thread = threading.Thread(target=self._processing_loop, name="ProcessingThread")
        processing_thread.start()
        
        frame_id = 0
        last_progress_log = 0
        
        while True:
            if self.stop_event.is_set() and self.frame_queue.empty():
                break
            
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    logging.info("End of video file reached.")
                    self.stop_event.set()
                    time.sleep(2) # Give processing thread time to finish
                    break
                
                try:
                    self.frame_queue.put(frame, timeout=1)
                    frame_id += 1
                    
                    # Progress logging every 100 frames
                    if frame_id - last_progress_log >= 100:
                        logging.info(f"Processed {frame_id} frames")
                        last_progress_log = frame_id
                        
                except queue.Full:
                    time.sleep(0.01)
            else:
                time.sleep(0.01) # Prevent busy-waiting if queue is full

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("'q' pressed, shutting down.")
                self.stop_event.set()
                break
        
        logging.info("Waiting for processing thread to complete...")
        processing_thread.join()
        self.cleanup()

    def cleanup(self):
        """Saves state and exports data upon completion."""
        logging.info("Cleaning up, saving state, and exporting data...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.save_state()
        
        if not self.results_data:
            logging.warning("No data was processed to export.")
            return
        
        logging.info(f"Normalizing and exporting {len(self.results_data)} frames of data...")
        
        # Flatten the data structure for CSV export
        export_data = []
        for frame_data in self.results_data:
            frame_id = frame_data['frame_id']
            actions = frame_data.get('actions', {})
            event = actions.get('event', 'None')
            
            if frame_data.get('objects'):
                for obj in frame_data['objects']:
                    export_data.append({
                        'frame_id': frame_id,
                        'event': event,
                        'object_id': obj.get('id', 'Unknown'),
                        'object_type': obj.get('type', 'Unknown'),
                        'team': obj.get('team', 'Unknown'),
                        'bbox_x1': obj.get('bbox_video', [0, 0, 0, 0])[0],
                        'bbox_y1': obj.get('bbox_video', [0, 0, 0, 0])[1],
                        'bbox_x2': obj.get('bbox_video', [0, 0, 0, 0])[2],
                        'bbox_y2': obj.get('bbox_video', [0, 0, 0, 0])[3],
                        'pitch_x': obj.get('pos_pitch', [0, 0])[0] if obj.get('pos_pitch') else 0,
                        'pitch_y': obj.get('pos_pitch', [0, 0])[1] if obj.get('pos_pitch') else 0,
                    })
            else:
                # Frame with no objects
                export_data.append({
                    'frame_id': frame_id,
                    'event': event,
                    'object_id': 'None',
                    'object_type': 'None',
                    'team': 'None',
                    'bbox_x1': 0, 'bbox_y1': 0, 'bbox_x2': 0, 'bbox_y2': 0,
                    'pitch_x': 0, 'pitch_y': 0,
                })
        
        df = pd.DataFrame(export_data)
        df.to_csv(self.config.OUTPUT_CSV_PATH, index=False)
        logging.info(f"Data successfully exported to {self.config.OUTPUT_CSV_PATH}")


if __name__ == '__main__':
    try:
        config = Config()
        
        # Validate video file exists
        if not os.path.exists(config.VIDEO_PATH):
            logging.error(f"Video file not found: {config.VIDEO_PATH}")
            print("Please update VIDEO_PATH in Config class to point to a valid video file.")
            exit(1)
        
        # Validate output directory is writable
        output_dir = os.path.dirname(config.OUTPUT_CSV_PATH) or '.'
        if not os.access(output_dir, os.W_OK):
            logging.error(f"Output directory is not writable: {output_dir}")
            exit(1)
        
        processor = VideoProcessor(config)
        processor.start()
        logging.info("Processing complete.")
        
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user.")
    except Exception as e:
        logging.critical(f"Critical error: {e}", exc_info=True)
        print(f"Error: {e}")
        exit(1)
