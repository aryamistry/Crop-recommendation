"""
Inference Module for Crop Recommendation System
Author: Principal ML Engineer
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CropRecommendationSystem:
    """
    Production inference system for crop recommendations.
    """
    
    def __init__(
        self,
        model_path: str = 'models/best_model.pkl',
        preprocessor_path: str = 'models/preprocessor.pkl'
    ):
        """
        Initialize inference system.
        
        Args:
            model_path: Path to trained model
            preprocessor_path: Path to fitted preprocessor
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        
        logger.info("CropRecommendationSystem initialized")
    
    def load_models(self) -> None:
        """Load trained model and preprocessor from disk."""
        try:
            # Load model
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
            
            # Load preprocessor
            if not Path(self.preprocessor_path).exists():
                raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")
            
            self.preprocessor = joblib.load(self.preprocessor_path)
            logger.info(f"Preprocessor loaded from {self.preprocessor_path}")
            
            self.is_loaded = True
            logger.info("All components loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def predict(
        self,
        input_data: Union[Dict[str, float], pd.DataFrame],
        return_probabilities: bool = False
    ) -> Union[str, Tuple[str, np.ndarray]]:
        """
        Make crop recommendation for given input.
        
        Args:
            input_data: Dictionary or DataFrame with feature values
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Recommended crop name, optionally with probabilities
        """
        if not self.is_loaded:
            self.load_models()
        
        # Convert dict to DataFrame if necessary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Validate input
        expected_features = self.preprocessor.feature_names
        if list(input_df.columns) != expected_features:
            # Reorder columns to match expected order
            try:
                input_df = input_df[expected_features]
            except KeyError as e:
                raise ValueError(f"Missing required features: {e}")
        
        # Preprocess input
        X_transformed, _ = self.preprocessor.transform(input_df)
        
        # Make prediction
        y_pred_encoded = self.model.predict(X_transformed)
        y_pred = self.preprocessor.inverse_transform_target(y_pred_encoded)
        
        # Get probabilities if requested
        if return_probabilities:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_transformed)
                return y_pred[0], probabilities[0]
            else:
                logger.warning("Model doesn't support probability predictions")
                return y_pred[0], None
        
        return y_pred[0]
    
    def predict_batch(
        self,
        input_data: pd.DataFrame,
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions for multiple samples.
        
        Args:
            input_data: DataFrame with feature values
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Array of recommended crops, optionally with probabilities
        """
        if not self.is_loaded:
            self.load_models()
        
        # Validate and preprocess
        expected_features = self.preprocessor.feature_names
        input_df = input_data[expected_features].copy()
        
        X_transformed, _ = self.preprocessor.transform(input_df)
        
        # Make predictions
        y_pred_encoded = self.model.predict(X_transformed)
        y_pred = self.preprocessor.inverse_transform_target(y_pred_encoded)
        
        if return_probabilities:
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_transformed)
                return y_pred, probabilities
            else:
                return y_pred, None
        
        return y_pred
    
    def get_top_n_recommendations(
        self,
        input_data: Dict[str, float],
        n: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get top N crop recommendations with probabilities.
        
        Args:
            input_data: Dictionary with feature values
            n: Number of top recommendations to return
            
        Returns:
            List of (crop_name, probability) tuples
        """
        if not self.is_loaded:
            self.load_models()
        
        if not hasattr(self.model, 'predict_proba'):
            logger.warning("Model doesn't support probability predictions")
            prediction = self.predict(input_data)
            return [(prediction, 1.0)]
        
        # Make prediction with probabilities
        _, probabilities = self.predict(input_data, return_probabilities=True)
        
        # Get class names
        classes = self.preprocessor.classes_
        
        # Get top N indices
        top_n_indices = np.argsort(probabilities)[-n:][::-1]
        
        # Create recommendations list
        recommendations = [
            (classes[idx], probabilities[idx])
            for idx in top_n_indices
        ]
        
        return recommendations
    
    def explain_prediction(
        self,
        input_data: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Provide explanation for prediction.
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Dictionary with prediction and explanation
        """
        if not self.is_loaded:
            self.load_models()
        
        # Make prediction
        prediction, probabilities = self.predict(
            input_data,
            return_probabilities=True
        )
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = dict(zip(
                self.preprocessor.feature_names,
                self.model.feature_importances_
            ))
        
        # Get top recommendations
        top_recommendations = self.get_top_n_recommendations(input_data, n=3)
        
        explanation = {
            'input_features': input_data,
            'recommended_crop': prediction,
            'confidence': probabilities[np.argmax(probabilities)] if probabilities is not None else None,
            'top_3_recommendations': top_recommendations,
            'feature_importance': feature_importance
        }
        
        return explanation
    
    def validate_input(self, input_data: Dict[str, float]) -> Tuple[bool, str]:
        """
        Validate input data before prediction.
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Tuple of (is_valid, message)
        """
        required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # Check all features present
        missing = set(required_features) - set(input_data.keys())
        if missing:
            return False, f"Missing features: {missing}"
        
        # Validate ranges
        validations = {
            'N': (0, 150, 'Nitrogen (N)'),
            'P': (0, 150, 'Phosphorus (P)'),
            'K': (0, 210, 'Potassium (K)'),
            'temperature': (-10, 50, 'Temperature'),
            'humidity': (0, 100, 'Humidity'),
            'ph': (0, 14, 'pH'),
            'rainfall': (0, 300, 'Rainfall')
        }
        
        for feature, (min_val, max_val, display_name) in validations.items():
            value = input_data[feature]
            if not isinstance(value, (int, float)):
                return False, f"{display_name} must be numeric"
            if not (min_val <= value <= max_val):
                return False, f"{display_name} must be between {min_val} and {max_val}"
        
        return True, "Valid input"


def create_sample_input() -> Dict[str, float]:
    """Create sample input for testing."""
    return {
        'N': 90,
        'P': 42,
        'K': 43,
        'temperature': 20.87,
        'humidity': 82.0,
        'ph': 6.5,
        'rainfall': 202.9
    }


def demo_inference():
    """Demonstrate inference capabilities."""
    logger.info("Running inference demo...")
    
    # Initialize system
    recommender = CropRecommendationSystem()
    
    try:
        # Load models
        recommender.load_models()
        
        # Create sample input
        sample_input = create_sample_input()
        logger.info(f"\nSample Input: {sample_input}")
        
        # Make prediction
        prediction = recommender.predict(sample_input)
        logger.info(f"Recommended Crop: {prediction}")
        
        # Get top 3 recommendations
        top_3 = recommender.get_top_n_recommendations(sample_input, n=3)
        logger.info("\nTop 3 Recommendations:")
        for i, (crop, prob) in enumerate(top_3, 1):
            logger.info(f"{i}. {crop}: {prob:.4f}")
        
        # Get explanation
        explanation = recommender.explain_prediction(sample_input)
        logger.info(f"\nConfidence: {explanation['confidence']:.4f}")
        
    except FileNotFoundError:
        logger.warning("Models not found. Train models first using training.py")
    except Exception as e:
        logger.error(f"Error during demo: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_inference()
