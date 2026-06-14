"""
Data Preprocessing Pipeline for Crop Recommendation System
Author: Principal ML Engineer
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import logging

logger = logging.getLogger(__name__)


class CropDataPreprocessor:
    """
    Production-grade preprocessing pipeline for crop recommendation data.
    Handles scaling, encoding, and data validation.
    """
    
    def __init__(self, scale_features: bool = True, random_state: int = 42):
        """
        Initialize preprocessor.
        
        Args:
            scale_features: Whether to apply feature scaling
            random_state: Random seed for reproducibility
        """
        self.scale_features = scale_features
        self.random_state = random_state
        
        # Initialize transformers
        self.scaler = StandardScaler() if scale_features else None
        self.label_encoder = LabelEncoder()
        
        # Store fitted state
        self.is_fitted = False
        self.feature_names = None
        self.target_name = None
        self.classes_ = None
        
        logger.info(f"Preprocessor initialized (scaling={scale_features})")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'CropDataPreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            self: Fitted preprocessor
        """
        logger.info("Fitting preprocessor...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        self.target_name = y.name if hasattr(y, 'name') else 'label'
        
        # Fit scaler
        if self.scaler is not None:
            self.scaler.fit(X)
            logger.info("Feature scaler fitted")
        
        # Fit label encoder
        self.label_encoder.fit(y)
        self.classes_ = self.label_encoder.classes_
        logger.info(f"Label encoder fitted with {len(self.classes_)} classes")
        
        self.is_fitted = True
        return self
    
    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Feature DataFrame
            y: Optional target Series
            
        Returns:
            Tuple of (X_transformed, y_transformed)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Validate features
        if list(X.columns) != self.feature_names:
            raise ValueError(f"Feature mismatch. Expected: {self.feature_names}")
        
        # Transform features
        if self.scaler is not None:
            X_transformed = self.scaler.transform(X)
        else:
            X_transformed = X.values
        
        # Transform target if provided
        y_transformed = None
        if y is not None:
            y_transformed = self.label_encoder.transform(y)
        
        return X_transformed, y_transformed
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Tuple of (X_transformed, y_transformed)
        """
        return self.fit(X, y).transform(X, y)
    
    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Convert encoded labels back to original labels.
        
        Args:
            y_encoded: Encoded label array
            
        Returns:
            Original labels
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse transform")
        
        return self.label_encoder.inverse_transform(y_encoded)
    
    def save(self, filepath: str = 'models/preprocessor.pkl') -> None:
        """
        Save fitted preprocessor to disk.
        
        Args:
            filepath: Path to save preprocessor
        """
        if not self.is_fitted:
            logger.warning("Saving unfitted preprocessor")
        
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str = 'models/preprocessor.pkl') -> 'CropDataPreprocessor':
        """
        Load fitted preprocessor from disk.
        
        Args:
            filepath: Path to preprocessor file
            
        Returns:
            Loaded preprocessor
        """
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor
    
    def get_config(self) -> Dict:
        """Get preprocessor configuration."""
        return {
            'scale_features': self.scale_features,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'n_classes': len(self.classes_) if self.classes_ is not None else 0,
            'classes': self.classes_.tolist() if self.classes_ is not None else [],
            'is_fitted': self.is_fitted
        }


def prepare_data(
    df: pd.DataFrame,
    target_col: str = 'label',
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, CropDataPreprocessor]:
    """
    Prepare data for model training with preprocessing.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Proportion of data for testing
        random_state: Random seed
        scale_features: Whether to scale features
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    logger.info("Preparing data...")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    logger.info(f"Features: {feature_cols}")
    logger.info(f"Target: {target_col} with {y.nunique()} classes")
    
    # Split data (stratified to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Initialize and fit preprocessor
    preprocessor = CropDataPreprocessor(
        scale_features=scale_features,
        random_state=random_state
    )
    
    # Fit on training data only (prevent data leakage)
    X_train_transformed, y_train_transformed = preprocessor.fit_transform(
        X_train, y_train
    )
    
    # Transform test data
    X_test_transformed, y_test_transformed = preprocessor.transform(
        X_test, y_test
    )
    
    logger.info("Data preparation completed")
    
    return (
        X_train_transformed,
        X_test_transformed,
        y_train_transformed,
        y_test_transformed,
        preprocessor
    )


def check_data_quality(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive data quality checks.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    logger.info("Performing data quality checks...")
    
    quality_report = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
    }
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    quality_report['infinite_values'] = inf_counts
    
    # Data type summary
    quality_report['dtypes'] = df.dtypes.astype(str).to_dict()
    
    logger.info(f"Quality check complete: {quality_report['duplicate_rows']} duplicates found")
    
    return quality_report


def remove_duplicates(df: pd.DataFrame, keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        keep: Which duplicates to keep ('first', 'last', False)
        
    Returns:
        DataFrame with duplicates removed
    """
    n_before = len(df)
    df_clean = df.drop_duplicates(keep=keep)
    n_after = len(df_clean)
    n_removed = n_before - n_after
    
    if n_removed > 0:
        logger.info(f"Removed {n_removed} duplicate rows ({n_removed/n_before*100:.2f}%)")
    else:
        logger.info("No duplicate rows found")
    
    return df_clean


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: List[str],
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Detect outliers using IQR method.
    
    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        threshold: IQR multiplier (typically 1.5 or 3.0)
        
    Returns:
        DataFrame with boolean columns indicating outliers
    """
    outlier_df = pd.DataFrame(index=df.index)
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_df[f'{col}_outlier'] = outliers
        
        n_outliers = outliers.sum()
        if n_outliers > 0:
            logger.info(f"{col}: {n_outliers} outliers detected ({n_outliers/len(df)*100:.2f}%)")
    
    return outlier_df


if __name__ == "__main__":
    # Test preprocessing module
    logging.basicConfig(level=logging.INFO)
    logger.info("Preprocessing module loaded successfully")
