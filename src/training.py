"""
Model Training Module for Crop Recommendation System
Author: Principal ML Engineer
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
import time
import logging

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import xgboost as xgb
import joblib

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Production-grade model training pipeline with hyperparameter optimization.
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize model trainer.
        
        Args:
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 uses all cores)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        logger.info(f"ModelTrainer initialized (random_state={random_state})")
    
    def get_base_models(self) -> Dict[str, Any]:
        """
        Get dictionary of base models with default parameters.
        
        Returns:
            Dictionary of model name to model instance
        """
        models = {
            'RandomForest': RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            'ExtraTrees': ExtraTreesClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            'GradientBoosting': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                eval_metric='mlogloss'
            ),
            'LogisticRegression': LogisticRegression(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                max_iter=1000
            )
        }
        
        return models
    
    def get_param_distributions(self) -> Dict[str, Dict]:
        """
        Get hyperparameter distributions for RandomizedSearchCV.
        
        Returns:
            Dictionary of model name to parameter distributions
        """
        param_distributions = {
            'RandomForest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            },
            'ExtraTrees': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 0.9, 1.0]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2]
            },
            'LogisticRegression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'saga']
            }
        }
        
        return param_distributions
    
    def train_with_cv(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Dict]:
        """
        Train all models with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of results for each model
        """
        logger.info(f"Training models with {cv}-fold cross-validation...")
        
        models = self.get_base_models()
        cv_splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )
        
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            start_time = time.time()
            
            # Cross-validation scores
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv_splitter,
                scoring='accuracy',
                n_jobs=self.n_jobs
            )
            
            # Fit on full training data
            model.fit(X_train, y_train)
            
            # Store model and results
            self.models[name] = model
            elapsed_time = time.time() - start_time
            
            self.results[name] = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': elapsed_time
            }
            
            logger.info(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            logger.info(f"{name} - Training Time: {elapsed_time:.2f}s")
        
        return self.results
    
    def hyperparameter_tuning(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_names: List[str] = None,
        n_iter: int = 50,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_names: List of model names to tune (None for all)
            n_iter: Number of iterations for random search
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of best parameters for each model
        """
        logger.info(f"\nPerforming hyperparameter tuning (n_iter={n_iter})...")
        
        base_models = self.get_base_models()
        param_distributions = self.get_param_distributions()
        
        if model_names is None:
            model_names = list(base_models.keys())
        
        tuned_models = {}
        best_params = {}
        
        cv_splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )
        
        for name in model_names:
            if name not in base_models:
                logger.warning(f"Model {name} not found, skipping...")
                continue
            
            logger.info(f"\nTuning {name}...")
            start_time = time.time()
            
            # Perform randomized search
            random_search = RandomizedSearchCV(
                estimator=base_models[name],
                param_distributions=param_distributions[name],
                n_iter=n_iter,
                cv=cv_splitter,
                scoring='accuracy',
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=0
            )
            
            random_search.fit(X_train, y_train)
            
            # Store results
            tuned_models[name] = random_search.best_estimator_
            best_params[name] = random_search.best_params_
            elapsed_time = time.time() - start_time
            
            logger.info(f"{name} - Best CV Score: {random_search.best_score_:.4f}")
            logger.info(f"{name} - Best Params: {random_search.best_params_}")
            logger.info(f"{name} - Tuning Time: {elapsed_time:.2f}s")
            
            # Update stored model with tuned version
            self.models[name] = tuned_models[name]
            if name in self.results:
                self.results[name]['best_params'] = best_params[name]
                self.results[name]['best_cv_score'] = random_search.best_score_
                self.results[name]['tuning_time'] = elapsed_time
        
        return best_params
    
    def evaluate_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Evaluate all trained models on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        logger.info("\nEvaluating models on test set...")
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            logger.info(f"\nEvaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            evaluation_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'predictions': y_pred
            }
            
            logger.info(f"{name} Test Metrics:")
            logger.info(f"  Accuracy:  {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall:    {recall:.4f}")
            logger.info(f"  F1 Score:  {f1:.4f}")
        
        return evaluation_results
    
    def select_best_model(
        self,
        evaluation_results: Dict[str, Dict],
        metric: str = 'accuracy'
    ) -> Tuple[str, Any]:
        """
        Select the best model based on specified metric.
        
        Args:
            evaluation_results: Evaluation results dictionary
            metric: Metric to use for selection
            
        Returns:
            Tuple of (best_model_name, best_model)
        """
        best_score = -np.inf
        best_name = None
        
        for name, results in evaluation_results.items():
            score = results.get(metric, -np.inf)
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        logger.info(f"\nBest model: {best_name} ({metric}={best_score:.4f})")
        
        return best_name, self.best_model
    
    def save_model(
        self,
        model_name: str = None,
        filepath: str = None
    ) -> None:
        """
        Save trained model to disk.
        
        Args:
            model_name: Name of model to save (None for best model)
            filepath: Path to save model (None for default)
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model selected. Run select_best_model first.")
            model_name = self.best_model_name
            model = self.best_model
        else:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        
        if filepath is None:
            filepath = f'models/{model_name.lower()}_model.pkl'
        
        joblib.dump(model, filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def get_feature_importance(
        self,
        model_name: str = None,
        feature_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Get feature importance from tree-based model.
        
        Args:
            model_name: Name of model (None for best model)
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError("No best model selected")
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} doesn't support feature importance")
            return None
        
        importance = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Training module loaded successfully")
