"""
Video processing utilities for sports analytics system.

This module provides common video processing functions and utilities.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional


def validate_video_file(video_path: str) -> bool:
    """
    Validate if a video file exists and is readable.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        True if video is valid, False otherwise
    """
    try:
        if not video_path:
            return False
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
            
        # Try to read one frame
        ret, _ = cap.read()
        cap.release()
        
        return ret
    except Exception as e:
        logging.error(f"Video validation failed: {e}")
        return False


def get_video_properties(video_path: str) -> dict:
    """
    Get video properties like FPS, frame count, resolution.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with video properties
    """
    properties = {
        'fps': 0,
        'frame_count': 0,
        'width': 0,
        'height': 0,
        'duration': 0
    }
    
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            properties['fps'] = cap.get(cv2.CAP_PROP_FPS)
            properties['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            properties['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            properties['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if properties['fps'] > 0:
                properties['duration'] = properties['frame_count'] / properties['fps']
                
        cap.release()
    except Exception as e:
        logging.error(f"Failed to get video properties: {e}")
        
    return properties


def resize_frame_if_needed(frame: np.ndarray, max_width: int = 1920, max_height: int = 1080) -> Tuple[np.ndarray, float]:
    """
    Resize frame if it exceeds maximum dimensions.
    
    Args:
        frame: Input frame
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Tuple of (resized_frame, scale_factor)
    """
    if frame is None or frame.size == 0:
        return frame, 1.0
        
    height, width = frame.shape[:2]
    
    if height <= max_height and width <= max_width:
        return frame, 1.0
        
    # Calculate scale factor
    scale_factor = min(max_width / width, max_height / height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    resized_frame = cv2.resize(frame, (new_width, new_height))
    return resized_frame, scale_factor


def create_video_hash(video_path: str) -> str:
    """
    Create a hash for a video file based on its properties.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Hash string
    """
    import hashlib
    import os
    
    try:
        # Use file size and modification time for hash
        stat = os.stat(video_path)
        hash_input = f"{video_path}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]
    except Exception as e:
        logging.error(f"Failed to create video hash: {e}")
        return "unknown"


def extract_frame_at_time(video_path: str, time_seconds: float) -> Optional[np.ndarray]:
    """
    Extract a frame at a specific time from video.
    
    Args:
        video_path: Path to the video file
        time_seconds: Time in seconds
        
    Returns:
        Frame as numpy array or None if failed
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(time_seconds * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None
    except Exception as e:
        logging.error(f"Failed to extract frame at time {time_seconds}: {e}")
        return None
