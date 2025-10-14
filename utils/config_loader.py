"""
Configuration Loader
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file (default: configs/default_config.yaml)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default config path
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(script_dir, "configs", "default_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
