"""
Logging utilities for sports analytics system.

This module provides centralized logging configuration and utilities
for the sports analytics application.
"""

import os
import logging
import warnings


def setup_logging(log_file: str = 'sports_analytics.log', level: int = logging.INFO):
    """
    Set up logging configuration for the sports analytics system.
    
    Args:
        log_file: Path to the log file
        level: Logging level
    """
    # Professional logging setup to provide clear, timestamped updates
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )


def setup_warnings():
    """Set up warning filters to suppress unnecessary warnings."""
    # Suppress unnecessary warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    # Suppress MPS pin_memory warning on Apple Silicon
    warnings.filterwarnings("ignore", message=".*pin_memory.*not supported on MPS.*")
    # Suppress other MPS-related warnings
    warnings.filterwarnings("ignore", message=".*MPS.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")


def setup_environment():
    """Set up environment variables for optimal performance."""
    # Set environment variable to disable pin_memory on MPS (Apple Silicon)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
