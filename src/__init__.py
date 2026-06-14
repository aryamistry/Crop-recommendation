"""
Crop Recommendation System - Production Package
Author: Principal ML Engineer

This package provides a complete, production-grade solution for crop recommendation
based on soil nutrients and environmental conditions.

Modules:
    utils: Utility functions and helpers
    preprocessing: Data preprocessing and transformation
    training: Model training and hyperparameter optimization
    inference: Production inference system

Example:
    >>> from src.inference import CropRecommendationSystem
    >>> recommender = CropRecommendationSystem()
    >>> recommender.load_models()
    >>> prediction = recommender.predict({'N': 90, 'P': 42, ...})
    >>> print(f"Recommended: {prediction}")
"""

__version__ = "1.0.0"
__author__ = "Principal ML Engineer"
__email__ = ""  # Add your email

# Import main classes for easier access
from .inference import CropRecommendationSystem
from .training import ModelTrainer
from .preprocessing import CropDataPreprocessor, prepare_data
from .utils import set_random_seeds, load_dataset

__all__ = [
    'CropRecommendationSystem',
    'ModelTrainer',
    'CropDataPreprocessor',
    'prepare_data',
    'set_random_seeds',
    'load_dataset'
]
