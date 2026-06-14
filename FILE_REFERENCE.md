# File Reference Guide
## Complete File Structure and Descriptions

---

## 📁 Project Structure Overview

```
crop_rec/
├── data/                          # Data directory (placeholder)
├── models/                        # Saved models (generated after training)
├── notebooks/                     # Jupyter notebooks
├── reports/                       # Analysis reports and figures
│   └── figures/                   # Visualization outputs
├── src/                           # Source code modules
└── [configuration and script files]
```

---

## 🐍 Source Code Modules (`src/`)

### `src/__init__.py`
**Purpose**: Package initialization  
**Size**: ~50 lines  
**Key Features**:
- Package version and metadata
- Import shortcuts for main classes
- `__all__` definition for clean imports

**Usage**:
```python
from src import CropRecommendationSystem, ModelTrainer
```

---

### `src/utils.py`
**Purpose**: Utility functions and helpers  
**Size**: ~350 lines  
**Key Components**:
- `set_random_seeds()` - Reproducibility
- `load_dataset()` - Data loading with validation
- `save_model_metadata()` / `load_model_metadata()` - Model info persistence
- `calculate_memory_usage()` - Memory profiling
- `get_feature_statistics()` - Statistical summaries
- `validate_input_data()` - Input validation
- `PerformanceTimer` - Execution timing context manager

**Usage**:
```python
from src.utils import set_random_seeds, load_dataset
set_random_seeds(42)
df = load_dataset('Crop-recommendation.csv')
```

---

### `src/preprocessing.py`
**Purpose**: Data preprocessing pipeline  
**Size**: ~450 lines  
**Key Components**:
- `CropDataPreprocessor` class - Main preprocessing pipeline
  - `fit()` - Learn preprocessing parameters
  - `transform()` - Apply preprocessing
  - `fit_transform()` - Combined operation
  - `save()` / `load()` - Persistence
- `prepare_data()` - Complete data preparation
- `check_data_quality()` - Quality assurance
- `remove_duplicates()` - Data cleaning
- `detect_outliers_iqr()` - Outlier detection

**Usage**:
```python
from src.preprocessing import prepare_data
X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
```

---

### `src/training.py`
**Purpose**: Model training and optimization  
**Size**: ~550 lines  
**Key Components**:
- `ModelTrainer` class - Training pipeline
  - `get_base_models()` - Model definitions
  - `get_param_distributions()` - Hyperparameter spaces
  - `train_with_cv()` - Cross-validation training
  - `hyperparameter_tuning()` - RandomizedSearchCV optimization
  - `evaluate_models()` - Test set evaluation
  - `select_best_model()` - Best model selection
  - `save_model()` - Model persistence
  - `get_feature_importance()` - Feature ranking

**Models Supported**:
- Random Forest Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- Logistic Regression

**Usage**:
```python
from src.training import ModelTrainer
trainer = ModelTrainer(random_state=42)
results = trainer.train_with_cv(X_train, y_train)
```

---

### `src/inference.py`
**Purpose**: Production inference system  
**Size**: ~450 lines  
**Key Components**:
- `CropRecommendationSystem` class - Production inference
  - `load_models()` - Load trained artifacts
  - `predict()` - Single prediction
  - `predict_batch()` - Batch predictions
  - `get_top_n_recommendations()` - Top-N crops with confidence
  - `explain_prediction()` - Explainability features
  - `validate_input()` - Input validation
- `create_sample_input()` - Sample data generator
- `demo_inference()` - Demonstration function

**Usage**:
```python
from src.inference import CropRecommendationSystem
recommender = CropRecommendationSystem()
recommender.load_models()
crop = recommender.predict(input_data)
```

---

## 📓 Notebooks (`notebooks/`)

### `CropRecommendation_Complete.ipynb`
**Purpose**: Complete EDA and analysis (Part 1)  
**Size**: ~1000 lines  
**Sections**:
1. Setup and Data Loading
2. Exploratory Data Analysis
   - Dataset information
   - Data quality checks
   - Statistical summaries
   - Class distribution
   - Feature distributions
   - Correlation analysis
   - Outlier detection

**Usage**: Open in Jupyter and run all cells for comprehensive analysis

---

### `CropRecommendation_Part2.ipynb`
**Purpose**: Model training and evaluation (Part 2)  
**Size**: ~800 lines  
**Sections**:
3. Data Preprocessing
4. Model Training & Cross-Validation
5. Hyperparameter Optimization
6. Test Set Evaluation
7. Best Model Selection
8. Confusion Matrix Analysis
9. Feature Importance Analysis

**Usage**: Continuation of Part 1, run after training models

---

## 🚀 Execution Scripts

### `main.py`
**Purpose**: Complete pipeline execution  
**Size**: ~250 lines  
**Features**:
- End-to-end workflow
- Data loading → Preprocessing → Training → Evaluation → Inference
- Progress logging
- Error handling
- Results summary

**Usage**:
```bash
python main.py
```

**Output**:
- `models/best_model.pkl`
- `models/preprocessor.pkl`
- `models/model_metadata.json`
- `reports/feature_importance.csv`
- `crop_recommendation.log`

---

### `example_usage.py`
**Purpose**: API usage demonstrations  
**Size**: ~400 lines  
**Examples**:
1. Single Prediction
2. With Confidence Scores
3. Top N Recommendations
4. Batch Predictions
5. Input Validation
6. Detailed Explanation
7. Scenario Comparison

**Usage**:
```bash
python example_usage.py
```

---

### `test_system.py`
**Purpose**: Comprehensive system testing  
**Size**: ~350 lines  
**Tests**:
1. Dependencies (7 packages)
2. Project Structure (files/folders)
3. Data Loading
4. Preprocessing Pipeline
5. Model Training Functions
6. Inference System
7. Utility Functions

**Usage**:
```bash
python test_system.py
```

**Expected**: All tests pass ✓

---

### `generate_visualizations.py`
**Purpose**: Generate all analysis figures  
**Size**: ~300 lines  
**Generates**:
- Class distribution plot
- Feature distributions (7 features)
- Correlation matrix heatmap
- Outlier detection box plots
- Feature-by-crop violin plots
- Summary statistics table

**Usage**:
```bash
python generate_visualizations.py
```

---

### `extract_dataset.py`
**Purpose**: Dataset extraction helper  
**Size**: ~30 lines  
**Function**: Template for extracting dataset from document

---

## 📚 Documentation Files

### `README.md`
**Purpose**: Complete project documentation  
**Size**: ~800 lines  
**Sections**:
- Overview
- Features
- Dataset description
- Installation instructions
- Usage guide
- Model performance
- API reference
- Academic report components
- Future scope

**Audience**: All users (developers, researchers, end-users)

---

### `QUICKSTART.md`
**Purpose**: 5-minute quick start guide  
**Size**: ~300 lines  
**Focus**:
- Fast setup
- Essential commands
- Common use cases
- Quick troubleshooting

**Audience**: Users wanting immediate results

---

### `SETUP_INSTRUCTIONS.md`
**Purpose**: Detailed setup walkthrough  
**Size**: ~500 lines  
**Sections**:
- Prerequisites
- Step-by-step installation
- Dataset setup
- Verification procedures
- Training instructions
- Prediction examples
- Comprehensive troubleshooting

**Audience**: New users, system administrators

---

### `PROJECT_SUMMARY.md`
**Purpose**: Executive and technical summary  
**Size**: ~600 lines  
**Sections**:
- Executive summary
- Technical specifications
- Architecture diagrams
- Methodology
- Results and metrics
- Implementation details
- Academic contribution
- Business impact
- Future enhancements

**Audience**: Stakeholders, academic reviewers, technical leads

---

### `FILE_REFERENCE.md`
**Purpose**: This file - complete file guide  
**Audience**: Developers, contributors

---

## ⚙️ Configuration Files

### `requirements.txt`
**Purpose**: Python dependencies  
**Size**: ~15 lines  
**Packages**:
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- joblib >= 1.0.0
- jupyter >= 1.0.0
- ipykernel >= 6.0.0
- scipy >= 1.7.0

---

### `.gitignore`
**Purpose**: Git version control exclusions  
**Size**: ~60 lines  
**Excludes**:
- Python cache files
- Virtual environments
- Jupyter checkpoints
- Log files
- Temporary files
- OS-specific files

---

## 📊 Data Files

### `Crop-recommendation.csv`
**Purpose**: Main dataset  
**Size**: 2,200 rows × 8 columns  
**Format**: CSV
**Columns**:
- N, P, K (nutrients)
- temperature, humidity, ph, rainfall (environment)
- label (target crop)

**Source**: Provided in document

---

## 🎯 Generated Files (After Training)

### `models/best_model.pkl`
**Purpose**: Trained ML model  
**Size**: ~50 MB  
**Format**: Pickle (joblib)  
**Contains**: Best performing model (typically XGBoost)

---

### `models/preprocessor.pkl`
**Purpose**: Fitted preprocessing pipeline  
**Size**: ~100 KB  
**Format**: Pickle (joblib)  
**Contains**: Scaler, encoder, feature names

---

### `models/model_metadata.json`
**Purpose**: Model information  
**Size**: ~2 KB  
**Format**: JSON  
**Contains**:
- Model name
- Performance metrics
- Training configuration
- Feature names
- Class names

---

### `reports/feature_importance.csv`
**Purpose**: Feature ranking  
**Size**: ~500 bytes  
**Format**: CSV  
**Columns**: feature, importance

---

### `reports/model_evaluation.csv`
**Purpose**: Model comparison  
**Size**: ~1 KB  
**Format**: CSV  
**Columns**: Model, Accuracy, Precision, Recall, F1 Score

---

### `reports/summary_statistics.csv`
**Purpose**: Data statistics  
**Size**: ~2 KB  
**Format**: CSV  
**Statistics**: count, mean, std, min, max, quartiles, skewness, kurtosis

---

### `reports/figures/*.png`
**Purpose**: Visualization outputs  
**Count**: 8 figures  
**Format**: PNG (300 DPI)  
**Files**:
- `class_distribution.png` - Class balance
- `feature_distributions.png` - Feature histograms
- `correlation_matrix.png` - Feature correlations
- `outlier_detection.png` - Box plots
- `cv_results.png` - Cross-validation comparison
- `test_performance.png` - Test metrics
- `confusion_matrix.png` - Classification matrix
- `feature_importance.png` - Feature ranking

---

### `crop_recommendation.log`
**Purpose**: Execution logs  
**Size**: Variable  
**Format**: Text  
**Contains**: Timestamped execution events

---

## 📖 Quick Reference

### Essential Files for Running
```
✓ Required:
  - src/*.py (all modules)
  - Crop-recommendation.csv
  - requirements.txt
  - main.py

✗ Optional:
  - notebooks/*.ipynb
  - example_usage.py
  - test_system.py
  - documentation files
```

### Files Generated by System
```
After running main.py:
  - models/best_model.pkl
  - models/preprocessor.pkl
  - models/model_metadata.json
  - reports/feature_importance.csv
  - reports/figures/*.png (if notebook run)
  - crop_recommendation.log
```

### Development vs Production Files
```
Development:
  - notebooks/*.ipynb
  - test_system.py
  - generate_visualizations.py
  - All documentation files

Production:
  - src/ (all modules)
  - models/*.pkl
  - requirements.txt
  - Optional: main.py for retraining
```

---

## 📊 File Statistics

**Total Files**: 25+  
**Total Lines of Code**: 2,500+  
**Documentation Lines**: 1,500+  
**Total Size (before training)**: ~500 KB  
**Total Size (after training)**: ~55 MB  

---

## 🔄 File Dependencies

```
main.py
  ├── src/utils.py
  ├── src/preprocessing.py
  ├── src/training.py
  └── src/inference.py

example_usage.py
  └── src/inference.py
      └── models/*.pkl

notebooks/*.ipynb
  ├── src/ (all modules)
  └── Crop-recommendation.csv

test_system.py
  └── src/ (all modules)

generate_visualizations.py
  ├── pandas, matplotlib, seaborn
  └── Crop-recommendation.csv
```

---

## 📝 Notes

1. **All Python files include**:
   - Comprehensive docstrings
   - Type hints
   - Error handling
   - Logging

2. **All documentation files**:
   - Markdown formatted
   - Step-by-step guides
   - Code examples
   - Troubleshooting sections

3. **Generated files should not be committed** (except model artifacts if desired)

4. **Dataset file must be obtained** from the provided document

---

**File Reference Version**: 1.0.0  
**Last Updated**: June 14, 2026  
**Status**: Complete ✅
