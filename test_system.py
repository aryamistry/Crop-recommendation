"""
System Testing Script for Crop Recommendation System
Tests all components and validates functionality

Author: Principal ML Engineer
"""

import sys
import os
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_error(msg):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")

def print_header(msg):
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")


def test_dependencies():
    """Test if all required packages are installed."""
    print_header("TEST 1: Dependency Check")
    
    required_packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'xgboost': 'XGBoost',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'joblib': 'Joblib'
    }
    
    all_installed = True
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print_success(f"{name} installed")
        except ImportError:
            print_error(f"{name} not found")
            all_installed = False
    
    return all_installed


def test_project_structure():
    """Test if project structure is correct."""
    print_header("TEST 2: Project Structure")
    
    required_dirs = ['src', 'data', 'models', 'reports', 'notebooks']
    required_files = [
        'src/__init__.py',
        'src/utils.py',
        'src/preprocessing.py',
        'src/training.py',
        'src/inference.py',
        'requirements.txt',
        'README.md',
        'main.py'
    ]
    
    all_present = True
    
    # Check directories
    for directory in required_dirs:
        if Path(directory).exists():
            print_success(f"Directory: {directory}/")
        else:
            print_error(f"Directory missing: {directory}/")
            all_present = False
    
    # Check files
    for filepath in required_files:
        if Path(filepath).exists():
            print_success(f"File: {filepath}")
        else:
            print_error(f"File missing: {filepath}")
            all_present = False
    
    return all_present


def test_data_loading():
    """Test data loading functionality."""
    print_header("TEST 3: Data Loading")
    
    sys.path.append('src')
    
    try:
        from utils import load_dataset
        
        # Check if dataset exists
        if not Path('Crop-recommendation.csv').exists():
            print_warning("Dataset file 'Crop-recommendation.csv' not found")
            print_info("Please ensure the dataset is in the project root")
            return False
        
        # Try to load
        df = load_dataset('Crop-recommendation.csv', validate=True)
        print_success(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Validate columns
        required_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
        if all(col in df.columns for col in required_cols):
            print_success("All required columns present")
        else:
            print_error("Missing required columns")
            return False
        
        # Check for data
        if len(df) > 0:
            print_success(f"Dataset contains {len(df)} samples")
        else:
            print_error("Dataset is empty")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Data loading failed: {str(e)}")
        return False


def test_preprocessing():
    """Test preprocessing pipeline."""
    print_header("TEST 4: Preprocessing Pipeline")
    
    try:
        from preprocessing import CropDataPreprocessor
        import pandas as pd
        import numpy as np
        
        # Create dummy data
        dummy_data = pd.DataFrame({
            'N': [90, 85, 60],
            'P': [42, 58, 55],
            'K': [43, 41, 44],
            'temperature': [20.87, 21.77, 23.00],
            'humidity': [82.0, 80.32, 82.32],
            'ph': [6.5, 7.04, 7.84],
            'rainfall': [202.9, 226.66, 263.96]
        })
        dummy_labels = pd.Series(['rice', 'rice', 'rice'])
        
        # Test preprocessor
        preprocessor = CropDataPreprocessor(scale_features=True)
        print_success("Preprocessor initialized")
        
        # Fit and transform
        X_transformed, y_transformed = preprocessor.fit_transform(dummy_data, dummy_labels)
        print_success("Fit and transform successful")
        
        # Check output shape
        if X_transformed.shape == (3, 7):
            print_success(f"Output shape correct: {X_transformed.shape}")
        else:
            print_error(f"Unexpected output shape: {X_transformed.shape}")
            return False
        
        # Check if scaled
        if preprocessor.scale_features:
            if np.abs(X_transformed.mean()) < 0.1:  # Should be close to 0
                print_success("Features scaled correctly")
            else:
                print_warning("Features may not be properly scaled")
        
        return True
        
    except Exception as e:
        print_error(f"Preprocessing test failed: {str(e)}")
        return False


def test_model_training():
    """Test model training (quick test)."""
    print_header("TEST 5: Model Training (Quick Test)")
    
    try:
        from training import ModelTrainer
        import numpy as np
        
        # Create dummy data
        np.random.seed(42)
        X_train = np.random.randn(100, 7)
        y_train = np.random.randint(0, 5, 100)
        
        # Initialize trainer
        trainer = ModelTrainer(random_state=42)
        print_success("ModelTrainer initialized")
        
        # Test getting base models
        models = trainer.get_base_models()
        print_success(f"Base models loaded: {len(models)} models")
        
        # Test quick training (just one model)
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        print_success("Quick training test passed")
        
        return True
        
    except Exception as e:
        print_error(f"Training test failed: {str(e)}")
        return False


def test_inference():
    """Test inference system."""
    print_header("TEST 6: Inference System")
    
    try:
        from inference import CropRecommendationSystem, create_sample_input
        
        # Initialize system
        recommender = CropRecommendationSystem()
        print_success("Inference system initialized")
        
        # Test input validation
        sample_input = create_sample_input()
        is_valid, message = recommender.validate_input(sample_input)
        
        if is_valid:
            print_success(f"Input validation working: {message}")
        else:
            print_error(f"Input validation failed: {message}")
            return False
        
        # Test invalid input
        invalid_input = {'N': -10}  # Missing fields and invalid value
        is_valid, message = recommender.validate_input(invalid_input)
        
        if not is_valid:
            print_success("Invalid input detected correctly")
        else:
            print_error("Invalid input not detected")
            return False
        
        # Note: Can't test actual prediction without trained models
        if Path('models/best_model.pkl').exists() and Path('models/preprocessor.pkl').exists():
            print_info("Trained models found - full inference test possible")
            recommender.load_models()
            prediction = recommender.predict(sample_input)
            print_success(f"Prediction successful: {prediction}")
        else:
            print_warning("Trained models not found - skipping prediction test")
            print_info("Run 'python main.py' to train models first")
        
        return True
        
    except Exception as e:
        print_error(f"Inference test failed: {str(e)}")
        return False


def test_utilities():
    """Test utility functions."""
    print_header("TEST 7: Utility Functions")
    
    try:
        from utils import (
            set_random_seeds,
            calculate_memory_usage,
            validate_input_data,
            format_bytes
        )
        import pandas as pd
        
        # Test random seeds
        set_random_seeds(42)
        print_success("Random seeds set")
        
        # Test memory usage calculation
        dummy_df = pd.DataFrame({'A': range(100), 'B': range(100)})
        mem_usage = calculate_memory_usage(dummy_df)
        print_success(f"Memory usage calculation: {mem_usage['total_mb']} MB")
        
        # Test input validation
        test_input = {
            'N': 90, 'P': 42, 'K': 43,
            'temperature': 20.87, 'humidity': 82.0,
            'ph': 6.5, 'rainfall': 202.9
        }
        is_valid, msg = validate_input_data(
            test_input,
            ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        )
        
        if is_valid:
            print_success("Input validation utility working")
        else:
            print_error(f"Input validation failed: {msg}")
            return False
        
        # Test format bytes
        formatted = format_bytes(1024*1024)
        print_success(f"Format bytes: {formatted}")
        
        return True
        
    except Exception as e:
        print_error(f"Utilities test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}CROP RECOMMENDATION SYSTEM - SYSTEM TESTS{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Project Structure", test_project_structure),
        ("Data Loading", test_data_loading),
        ("Preprocessing", test_preprocessing),
        ("Model Training", test_model_training),
        ("Inference", test_inference),
        ("Utilities", test_utilities)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        if result:
            print_success(f"{test_name}")
        else:
            print_error(f"{test_name}")
    
    print(f"\n{Colors.BOLD}Results: {passed}/{total} tests passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED{Colors.END}")
        print(f"\n{Colors.GREEN}System is ready for use!{Colors.END}")
        print(f"\nNext steps:")
        print(f"  1. Ensure dataset is in place: Crop-recommendation.csv")
        print(f"  2. Run training: python main.py")
        print(f"  3. Try examples: python example_usage.py")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ SOME TESTS FAILED{Colors.END}")
        print(f"\n{Colors.YELLOW}Please fix the issues above before proceeding{Colors.END}")
        print(f"\nCommon solutions:")
        print(f"  • Install dependencies: pip install -r requirements.txt")
        print(f"  • Check project structure matches documentation")
        print(f"  • Ensure dataset file is present")
        return False


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Tests interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {str(e)}{Colors.END}")
        sys.exit(1)
