"""
Utility functions for Crop Recommendation System
Author: Principal ML Engineer
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seeds set to {seed} for reproducibility")


def create_directory_structure(base_path: str = '.') -> None:
    """
    Create project directory structure if it doesn't exist.
    
    Args:
        base_path: Base path for project structure
    """
    directories = [
        'data',
        'notebooks',
        'src',
        'models',
        'reports',
        'reports/figures'
    ]
    
    for directory in directories:
        path = Path(base_path) / directory
        path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Directory structure created successfully")


def load_dataset(filepath: str, validate: bool = True) -> pd.DataFrame:
    """
    Load and optionally validate the crop recommendation dataset.
    
    Args:
        filepath: Path to CSV file
        validate: Whether to perform validation checks
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If validation fails
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        if validate:
            required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
            missing_cols = set(required_columns) - set(df.columns)
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info("Dataset validation passed")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def save_model_metadata(
    metadata: Dict[str, Any],
    filepath: str = 'models/model_metadata.json'
) -> None:
    """
    Save model metadata to JSON file.
    
    Args:
        metadata: Dictionary containing model metadata
        filepath: Path to save metadata
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        logger.info(f"Model metadata saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")
        raise


def load_model_metadata(filepath: str = 'models/model_metadata.json') -> Dict[str, Any]:
    """
    Load model metadata from JSON file.
    
    Args:
        filepath: Path to metadata file
        
    Returns:
        Dictionary containing model metadata
    """
    try:
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Model metadata loaded from {filepath}")
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        raise


def calculate_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate memory usage of DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with memory usage statistics
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 ** 2)
    
    return {
        'total_bytes': memory_bytes,
        'total_mb': round(memory_mb, 2),
        'per_row_bytes': round(memory_bytes / len(df), 2)
    }


def get_feature_statistics(
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Get comprehensive statistics for numerical features.
    
    Args:
        df: Input DataFrame
        feature_cols: List of feature column names
        
    Returns:
        DataFrame with statistics
    """
    stats = df[feature_cols].describe().T
    stats['missing'] = df[feature_cols].isnull().sum()
    stats['missing_pct'] = (stats['missing'] / len(df)) * 100
    stats['unique'] = df[feature_cols].nunique()
    stats['skewness'] = df[feature_cols].skew()
    stats['kurtosis'] = df[feature_cols].kurtosis()
    
    return stats.round(4)


def validate_input_data(
    input_data: Dict[str, float],
    expected_features: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Validate input data for prediction.
    
    Args:
        input_data: Dictionary with feature values
        expected_features: List of expected feature names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check all features present
    missing_features = set(expected_features) - set(input_data.keys())
    if missing_features:
        return False, f"Missing features: {missing_features}"
    
    # Check for non-numeric values
    try:
        for key, value in input_data.items():
            if key in expected_features:
                float(value)
    except (ValueError, TypeError):
        return False, f"Invalid numeric value for feature: {key}"
    
    # Range validation (basic checks)
    range_checks = {
        'N': (0, 150),
        'P': (0, 150),
        'K': (0, 210),
        'temperature': (-10, 50),
        'humidity': (0, 100),
        'ph': (0, 14),
        'rainfall': (0, 300)
    }
    
    for feature, (min_val, max_val) in range_checks.items():
        if feature in input_data:
            value = input_data[feature]
            if not (min_val <= value <= max_val):
                return False, f"{feature} value {value} outside expected range [{min_val}, {max_val}]"
    
    return True, None


class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        import time
        self.start_time = time.time()
        logger.info(f"{self.name} started...")
        return self
        
    def __exit__(self, *args):
        import time
        elapsed = time.time() - self.start_time
        logger.info(f"{self.name} completed in {elapsed:.2f} seconds")


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., '1.5 MB')
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


if __name__ == "__main__":
    # Test utilities
    set_random_seeds(42)
    create_directory_structure()
    logger.info("Utilities module loaded successfully")
