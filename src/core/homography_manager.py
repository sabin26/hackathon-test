"""
Homography management module for sports analytics.

This module handles the calculation and application of homography transformations
to map video coordinates to pitch coordinates for accurate spatial analysis.
"""

import time
import logging
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional


class HomographyManager:
    """Manages homography with correct reprojection error calculation."""
    
    def __init__(self, pitch_dims: Tuple[float, float], segmentation_model):
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
