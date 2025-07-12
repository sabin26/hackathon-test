"""
Segmentation model module for sports analytics.

This module contains the segmentation model class that handles pitch line detection
using deep learning models or fallback to computer vision techniques.
"""

import os
import time
import logging
import gc
import cv2
import numpy as np
import torch
from torchvision import transforms
from contextlib import contextmanager


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
