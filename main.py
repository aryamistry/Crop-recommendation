"""
Main execution script for Crop Recommendation System
Demonstrates complete workflow from data loading to inference

Author: Principal ML Engineer
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from utils import (
    set_random_seeds,
    load_dataset,
    save_model_metadata,
    PerformanceTimer,
    create_directory_structure
)
from preprocessing import (
    prepare_data,
    check_data_quality,
    remove_duplicates
)
from training import ModelTrainer
from inference import CropRecommendationSystem, create_sample_input

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crop_recommendation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    
    print("=" * 80)
    print("CROP RECOMMENDATION SYSTEM - PRODUCTION PIPELINE")
    print("=" * 80)
    print()
    
    # Configuration
    RANDOM_STATE = 42
    DATA_PATH = 'Crop-recommendation.csv'
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    TUNING_ITERATIONS = 50
    
    # Set random seeds
    set_random_seeds(RANDOM_STATE)
    
    # Create directory structure
    create_directory_structure()
    
    # Step 1: Load and validate data
    print("\n" + "=" * 80)
    print("STEP 1: DATA LOADING AND VALIDATION")
    print("=" * 80)
    
    with PerformanceTimer("Data Loading"):
        try:
            df = load_dataset(DATA_PATH, validate=True)
            logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        except FileNotFoundError:
            logger.error(f"Dataset file not found: {DATA_PATH}")
            logger.info("Please ensure Crop-recommendation.csv is in the project root")
            logger.info("You can use the provided CSV data from the document")
            return
    
    # Data quality check
    quality_report = check_data_quality(df)
    logger.info(f"Data quality check: {quality_report['duplicate_rows']} duplicates")
    
    if quality_report['duplicate_rows'] > 0:
        df = remove_duplicates(df)
    
    print(f"\n✓ Dataset loaded and validated successfully")
    print(f"  - Samples: {len(df):,}")
    print(f"  - Features: {len(df.columns) - 1}")
    print(f"  - Classes: {df['label'].nunique()}")
    
    # Step 2: Data preprocessing
    print("\n" + "=" * 80)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 80)
    
    with PerformanceTimer("Data Preprocessing"):
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(
            df,
            target_col='label',
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            scale_features=True
        )
    
    print(f"\n✓ Data preprocessing completed")
    print(f"  - Training samples: {len(X_train):,}")
    print(f"  - Test samples: {len(X_test):,}")
    print(f"  - Features scaled: Yes")
    
    # Save preprocessor
    preprocessor.save('models/preprocessor.pkl')
    logger.info("Preprocessor saved to models/preprocessor.pkl")
    
    # Step 3: Model training with cross-validation
    print("\n" + "=" * 80)
    print("STEP 3: MODEL TRAINING (Cross-Validation)")
    print("=" * 80)
    
    trainer = ModelTrainer(random_state=RANDOM_STATE, n_jobs=-1)
    
    with PerformanceTimer("Model Training"):
        cv_results = trainer.train_with_cv(X_train, y_train, cv=CV_FOLDS)
    
    print(f"\n✓ Cross-validation completed ({CV_FOLDS} folds)")
    print("\nModel Comparison (CV Accuracy):")
    print("-" * 60)
    for model_name in sorted(cv_results.keys(), key=lambda x: cv_results[x]['cv_mean'], reverse=True):
        result = cv_results[model_name]
        print(f"  {model_name:20s}: {result['cv_mean']:.4f} (±{result['cv_std']:.4f}) | Time: {result['training_time']:.2f}s")
    
    # Step 4: Hyperparameter tuning (optional - can be skipped for faster execution)
    PERFORM_TUNING = True  # Set to False to skip tuning
    
    if PERFORM_TUNING:
        print("\n" + "=" * 80)
        print("STEP 4: HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        
        # Tune only top 3 models to save time
        top_models = sorted(
            cv_results.keys(),
            key=lambda x: cv_results[x]['cv_mean'],
            reverse=True
        )[:3]
        
        logger.info(f"Tuning models: {top_models}")
        
        with PerformanceTimer("Hyperparameter Tuning"):
            best_params = trainer.hyperparameter_tuning(
                X_train,
                y_train,
                model_names=top_models,
                n_iter=TUNING_ITERATIONS,
                cv=CV_FOLDS
            )
        
        print(f"\n✓ Hyperparameter tuning completed")
    else:
        logger.info("Skipping hyperparameter tuning (PERFORM_TUNING=False)")
    
    # Step 5: Model evaluation on test set
    print("\n" + "=" * 80)
    print("STEP 5: MODEL EVALUATION (Test Set)")
    print("=" * 80)
    
    with PerformanceTimer("Model Evaluation"):
        eval_results = trainer.evaluate_models(X_test, y_test)
    
    print("\nTest Set Performance:")
    print("-" * 60)
    for model_name in sorted(eval_results.keys(), key=lambda x: eval_results[x]['accuracy'], reverse=True):
        result = eval_results[model_name]
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1 Score:  {result['f1_score']:.4f}")
    
    # Step 6: Select and save best model
    print("\n" + "=" * 80)
    print("STEP 6: BEST MODEL SELECTION")
    print("=" * 80)
    
    best_name, best_model = trainer.select_best_model(eval_results, metric='accuracy')
    
    print(f"\n✓ Best Model: {best_name}")
    print(f"  Test Accuracy: {eval_results[best_name]['accuracy']:.4f}")
    
    # Save best model
    trainer.save_model(filepath='models/best_model.pkl')
    logger.info("Best model saved to models/best_model.pkl")
    
    # Get feature importance
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    importance_df = trainer.get_feature_importance(
        model_name=best_name,
        feature_names=feature_names
    )
    
    if importance_df is not None:
        print("\nFeature Importance:")
        print("-" * 60)
        for idx, row in importance_df.iterrows():
            print(f"  {row['feature']:15s}: {row['importance']:.4f}")
        
        importance_df.to_csv('reports/feature_importance.csv', index=False)
        logger.info("Feature importance saved to reports/feature_importance.csv")
    
    # Save model metadata
    metadata = {
        'model_name': best_name,
        'test_accuracy': float(eval_results[best_name]['accuracy']),
        'test_precision': float(eval_results[best_name]['precision']),
        'test_recall': float(eval_results[best_name]['recall']),
        'test_f1': float(eval_results[best_name]['f1_score']),
        'cv_accuracy_mean': float(cv_results[best_name]['cv_mean']),
        'cv_accuracy_std': float(cv_results[best_name]['cv_std']),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(feature_names),
        'n_classes': len(preprocessor.classes_),
        'classes': preprocessor.classes_.tolist(),
        'feature_names': feature_names,
        'random_state': RANDOM_STATE,
        'preprocessing': {
            'scaling': 'StandardScaler',
            'encoding': 'LabelEncoder'
        }
    }
    
    save_model_metadata(metadata, 'models/model_metadata.json')
    logger.info("Model metadata saved to models/model_metadata.json")
    
    # Step 7: Inference demonstration
    print("\n" + "=" * 80)
    print("STEP 7: INFERENCE DEMONSTRATION")
    print("=" * 80)
    
    recommender = CropRecommendationSystem(
        model_path='models/best_model.pkl',
        preprocessor_path='models/preprocessor.pkl'
    )
    
    recommender.load_models()
    
    # Test with sample input
    sample_input = create_sample_input()
    
    print("\nSample Input:")
    print("-" * 60)
    for feature, value in sample_input.items():
        print(f"  {feature:15s}: {value}")
    
    # Make prediction
    prediction, probabilities = recommender.predict(
        sample_input,
        return_probabilities=True
    )
    
    print(f"\n✓ Recommended Crop: {prediction}")
    print(f"  Confidence: {probabilities[np.argmax(probabilities)]:.2%}")
    
    # Get top 3 recommendations
    top_3 = recommender.get_top_n_recommendations(sample_input, n=3)
    
    print("\nTop 3 Recommendations:")
    print("-" * 60)
    for i, (crop, prob) in enumerate(top_3, 1):
        print(f"  {i}. {crop:15s}: {prob:.2%}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    print("\n📁 Generated Files:")
    print("  - models/best_model.pkl")
    print("  - models/preprocessor.pkl")
    print("  - models/model_metadata.json")
    print("  - reports/feature_importance.csv")
    print("  - crop_recommendation.log")
    
    print("\n📊 Key Results:")
    print(f"  - Best Model: {best_name}")
    print(f"  - Test Accuracy: {eval_results[best_name]['accuracy']:.4f}")
    print(f"  - CV Accuracy: {cv_results[best_name]['cv_mean']:.4f} (±{cv_results[best_name]['cv_std']:.4f})")
    
    print("\n✅ System is ready for production deployment!")
    print("\nNext Steps:")
    print("  1. Review the comprehensive Jupyter notebook: notebooks/CropRecommendation_Complete.ipynb")
    print("  2. Explore visualizations in: reports/figures/")
    print("  3. Test inference with: python -c \"from src.inference import demo_inference; demo_inference()\"")
    print("  4. Read documentation: README.md")
    
    logger.info("Main pipeline execution completed successfully")


if __name__ == "__main__":
    try:
        import numpy as np
        main()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease check:")
        print("  1. All dependencies are installed (run: pip install -r requirements.txt)")
        print("  2. Dataset file exists: Crop-recommendation.csv")
        print("  3. Check crop_recommendation.log for details")
        sys.exit(1)
