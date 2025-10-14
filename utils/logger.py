"""
Logging Configuration
"""

import logging
import os
from datetime import datetime


def setup_logging(config: dict = None):
    """
    Set up logging configuration.

    Args:
        config: Configuration dictionary with logging settings
    """
    if config is None:
        config = {
            'level': 'INFO',
            'log_dir': './logs/',
            'log_to_file': True,
            'log_to_console': True
        }

    # Create log directory if it doesn't exist
    log_dir = config.get('log_dir', './logs/')
    os.makedirs(log_dir, exist_ok=True)

    # Get log level
    level_str = config.get('level', 'INFO').upper()
    level = getattr(logging, level_str, logging.INFO)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if config.get('log_to_console', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if config.get('log_to_file', True):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'cda_agent_{timestamp}.log')

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    logger.info(f"Logging initialized at level: {level_str}")
