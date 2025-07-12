"""
Object tracking module for sports analytics.

This module contains the ObjectTracker class that handles YOLO-based object detection
and tracking for players and other objects in sports videos.
"""

import os
import time
import logging
import warnings
import cv2
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
from ultralytics import YOLO


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
