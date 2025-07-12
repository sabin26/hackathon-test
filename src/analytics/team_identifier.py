"""
Team identification module for sports analytics.

This module contains the TeamIdentifier class that uses machine learning
to automatically identify and classify teams based on player appearance.
"""

import time
import logging
import threading
import cv2
import numpy as np
from typing import Optional, List, Dict
from collections import defaultdict
from sklearn.cluster import KMeans


class TeamIdentifier:
    """Efficiently collects crops and automatically assigns Team A/Team B labels."""
    
    def __init__(self, n_clusters: int, config=None):
        self.n_clusters = n_clusters
        self.config = config  # Will be injected during initialization
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
            
            # Use config if available, otherwise use default values
            min_crop_size = 100
            team_feature_bins = 16
            if self.config and hasattr(self.config, 'MODEL_PARAMS'):
                min_crop_size = self.config.MODEL_PARAMS.get('MIN_CROP_SIZE', 100)
                team_feature_bins = self.config.MODEL_PARAMS.get('TEAM_FEATURE_BINS', 16)
            
            if crop_bgr.size < min_crop_size:
                return None
            
            # Convert to HSV for better color representation
            crop_hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            
            # Create more robust histogram
            bins = [team_feature_bins] * 2
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
                    if self.is_fitted and self.kmeans is not None:
                        cluster_id = self.kmeans.predict([features])[0]
                        max_crops = 15  # Default value
                        if self.config and hasattr(self.config, 'MODEL_PARAMS'):
                            max_crops = self.config.MODEL_PARAMS.get('MAX_CROPS_PER_CLUSTER', 15)
                        if len(self.example_crops[cluster_id]) < max_crops:
                            self.example_crops[cluster_id].append(crop_resized)
                else:
                    self.crop_samples.append(np.zeros((64, 64, 3), dtype=np.uint8))

            except Exception as e:
                logging.error(f"Crop storage failed: {e}")

    def fit(self) -> None:
        """Fits the K-Means model and organizes crops by cluster with improved stability."""
        if len(self.feature_samples) < self.n_clusters:
            logging.warning(f"Not enough samples ({len(self.feature_samples)}) for {self.n_clusters} clusters")
            return

        try:
            logging.info("Fitting team feature model...")

            # FIXED: Improve clustering stability with multiple runs
            best_inertia = float('inf')
            best_kmeans = None

            # Try multiple random initializations to find the most stable clustering
            for attempt in range(5):  # Try 5 different initializations
                temp_kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    n_init=10,  # Multiple initializations per attempt
                    random_state=42 + attempt,  # Different seed each time
                    max_iter=300
                )
                temp_kmeans.fit(self.feature_samples)

                if temp_kmeans.inertia_ < best_inertia:
                    best_inertia = temp_kmeans.inertia_
                    best_kmeans = temp_kmeans

            self.kmeans = best_kmeans
            self.is_fitted = True

            # Clear existing example crops
            self.example_crops.clear()

            # Organize existing crops by cluster
            for i, features in enumerate(self.feature_samples):
                if i < len(self.crop_samples) and self.kmeans is not None:
                    cluster_id = self.kmeans.predict([features])[0]
                    max_crops = 15  # Default value
                    if self.config and hasattr(self.config, 'MODEL_PARAMS'):
                        max_crops = self.config.MODEL_PARAMS.get('MAX_CROPS_PER_CLUSTER', 15)
                    if len(self.example_crops[cluster_id]) < max_crops:
                        self.example_crops[cluster_id].append(self.crop_samples[i])

            # FIXED: Validate clustering quality
            cluster_sizes = [len(self.example_crops[i]) for i in range(self.n_clusters)]
            logging.info(f"Team K-Means model fitted with {len(self.feature_samples)} samples.")
            logging.info(f"Cluster sizes: {cluster_sizes}, Inertia: {best_inertia:.2f}")

            # Warn if clusters are very unbalanced
            if max(cluster_sizes) > 3 * min(cluster_sizes):
                logging.warning("Unbalanced clusters detected - team identification may be less accurate")

        except Exception as e:
            logging.error(f"Model fitting failed: {e}")
            self.is_fitted = False

    def classify_player(self, frame: np.ndarray, bbox: List[int]) -> str:
        """Classifies a single player based on the fitted model and labeled map."""
        if not self.is_fitted or self.kmeans is None:
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
