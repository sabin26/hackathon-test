"""
Performance monitoring utilities for sports analytics system.

This module provides performance monitoring and optimization utilities.
"""

import time
import psutil
import logging
from typing import Dict, Any
from collections import deque
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    frames_processed: int = 0
    frames_skipped: int = 0
    processing_times: deque = None
    memory_usage: deque = None
    errors_count: int = 0
    
    def __post_init__(self):
        if self.processing_times is None:
            self.processing_times = deque(maxlen=100)
        if self.memory_usage is None:
            self.memory_usage = deque(maxlen=50)


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.start_time = time.time()
        
    def record_frame_processed(self, processing_time: float):
        """Record a processed frame with its processing time."""
        self.metrics.frames_processed += 1
        self.metrics.processing_times.append(processing_time)
        
    def record_frame_skipped(self):
        """Record a skipped frame."""
        self.metrics.frames_skipped += 1
        
    def record_error(self):
        """Record an error occurrence."""
        self.metrics.errors_count += 1
        
    def record_memory_usage(self):
        """Record current memory usage."""
        memory_percent = psutil.virtual_memory().percent
        self.metrics.memory_usage.append(memory_percent)
        
    def get_average_processing_time(self) -> float:
        """Get average processing time per frame."""
        if not self.metrics.processing_times:
            return 0.0
        return sum(self.metrics.processing_times) / len(self.metrics.processing_times)
        
    def get_current_fps(self) -> float:
        """Calculate current FPS based on recent processing times."""
        avg_time = self.get_average_processing_time()
        return 1.0 / avg_time if avg_time > 0 else 0.0
        
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        if not self.metrics.memory_usage:
            return 0.0
        return self.metrics.memory_usage[-1]
        
    def get_total_runtime(self) -> float:
        """Get total runtime in seconds."""
        return time.time() - self.start_time
        
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_frames = self.metrics.frames_processed + self.metrics.frames_skipped
        skip_rate = (self.metrics.frames_skipped / total_frames * 100) if total_frames > 0 else 0
        
        return {
            'frames_processed': self.metrics.frames_processed,
            'frames_skipped': self.metrics.frames_skipped,
            'total_frames': total_frames,
            'skip_rate_percent': skip_rate,
            'average_processing_time': self.get_average_processing_time(),
            'current_fps': self.get_current_fps(),
            'memory_usage_percent': self.get_memory_usage(),
            'errors_count': self.metrics.errors_count,
            'total_runtime': self.get_total_runtime()
        }
        
    def log_performance_summary(self):
        """Log performance summary."""
        summary = self.get_summary()
        logging.info("=== Performance Summary ===")
        logging.info(f"Frames processed: {summary['frames_processed']}")
        logging.info(f"Frames skipped: {summary['frames_skipped']} ({summary['skip_rate_percent']:.1f}%)")
        logging.info(f"Average processing time: {summary['average_processing_time']:.3f}s")
        logging.info(f"Current FPS: {summary['current_fps']:.1f}")
        logging.info(f"Memory usage: {summary['memory_usage_percent']:.1f}%")
        logging.info(f"Errors: {summary['errors_count']}")
        logging.info(f"Total runtime: {summary['total_runtime']:.1f}s")


def check_system_resources() -> Dict[str, Any]:
    """Check current system resource usage."""
    return {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'available_memory_gb': psutil.virtual_memory().available / (1024**3),
        'disk_usage_percent': psutil.disk_usage('/').percent
    }


def is_system_under_pressure(memory_threshold: float = 85.0, cpu_threshold: float = 90.0) -> bool:
    """Check if system is under resource pressure."""
    resources = check_system_resources()
    return (resources['memory_percent'] > memory_threshold or 
            resources['cpu_percent'] > cpu_threshold)


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        logging.debug(f"{self.name} took {elapsed:.3f} seconds")
        
    def elapsed(self) -> float:
        """Get elapsed time since timer started."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
