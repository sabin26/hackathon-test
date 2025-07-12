"""
Jersey number detection module for sports analytics.

This module contains the JerseyNumberDetector class that uses YOLO and OCR
to detect and recognize jersey numbers from player crops.
"""

import os
import time
import logging
import warnings
import cv2
import numpy as np
import easyocr
from typing import Optional, List
from ultralytics import YOLO


class JerseyNumberDetector:
    """Detects jersey numbers from player crops using YOLO and OCR."""

    def __init__(self, yolo_model_path: Optional[str] = None, config=None):
        self.yolo_model = None
        self.ocr_reader = None
        self.config = config  # Will be injected during initialization
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
            enable_ocr = True
            if self.config and hasattr(self.config, 'ENABLE_OCR'):
                enable_ocr = getattr(self.config, 'ENABLE_OCR', True)
            
            if not enable_ocr:
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

            # FIXED: Improved number extraction and validation
            detected_numbers = []
            for (_, text, confidence) in results:  # bbox not needed for this implementation
                try:
                    conf_value = float(confidence)
                except (ValueError, TypeError):
                    conf_value = 0.0

                if conf_value > self.ocr_confidence_threshold:
                    # Extract numeric characters with better filtering
                    numeric_text = ''.join(filter(str.isdigit, text.strip()))
                    if numeric_text:
                        try:
                            number = int(numeric_text)
                            # Validate jersey number range (typically 1-99)
                            if 1 <= number <= 99:
                                detected_numbers.append((number, conf_value))
                                logging.debug(f"Detected jersey number candidate: {number} (confidence: {conf_value:.2f})")
                        except ValueError:
                            continue

            # Return the number with highest confidence if multiple detected
            if detected_numbers:
                detected_numbers.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence
                best_number, best_confidence = detected_numbers[0]
                logging.debug(f"Selected jersey number: {best_number} (confidence: {best_confidence:.2f})")
                return best_number

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
