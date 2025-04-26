#!/usr/bin/env python3
# utils.py

import os
import logging
from typing import Dict, Any, Optional

def setup_logging(log_dir: str, log_name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to store log files
        log_name: Base name for the log file
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        
    Returns:
        Configured logger
    """
    os.makedirs(os.getcwd(), log_dir, exist_ok=True)
    
    log_file = os.path.join(os.getcwd(), log_dir, f"{log_name}.log")
    
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Log initial message
    logger.info(f"Logging initialized: {log_file}")
    
    return logger

def load_config_file(config_file: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_file: Path to the JSON configuration file
        
    Returns:
        Dictionary with configuration, or None if failed
    """
    import json
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration file {config_file}: {e}")
        return None

def validate_directory(directory: str, create: bool = True) -> bool:
    """
    Validate if a directory exists and create it if needed.
    
    Args:
        directory: Directory path to validate
        create: Whether to create the directory if it doesn't exist
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    if os.path.exists(directory):
        if os.path.isdir(directory):
            return True
        else:
            return False
    elif create:
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except:
            return False
    else:
        return False