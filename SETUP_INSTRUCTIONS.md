# Complete Setup Instructions
## Crop Recommendation System

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Dataset Setup](#dataset-setup)
4. [Verification](#verification)
5. [Training Models](#training-models)
6. [Making Predictions](#making-predictions)
7. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

### System Requirements
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: 2GB free space
- **Internet**: For package installation

### Check Python Version

```bash
python --version
# Should show Python 3.8.x or higher
```

If Python is not installed:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: `brew install python3` or download from python.org
- **Linux**: `sudo apt install python3 python3-pip`

---

## 2. Installation

### Step 1: Navigate to Project Directory

```bash
cd d:\crop_rec
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy (numerical computing)
- pandas (data manipulation)
- scikit-learn (machine learning)
- xgboost (gradient boosting)
- matplotlib (plotting)
- seaborn (visualization)
- joblib (model persistence)
- jupyter (notebooks)

**Installation time**: 2-5 minutes depending on internet speed

### Step 5: Verify Installation

```bash
python test_system.py
```

This runs comprehensive tests. You should see:
```
✓ ALL TESTS PASSED
System is ready for use!
```

---

## 3. Dataset Setup

### Option 1: Use Provided Dataset (Recommended)

The dataset from the document needs to be saved as `Crop-recommendation.csv`.

**Important**: The CSV file should be in the project root directory:
```
d:\crop_rec\Crop-recommendation.csv
```

**Format**:
```csv
N,P,K,temperature,humidity,ph,rainfall,label
90,42,43,20.87974371,82.00274423,6.502985292000001,202.9355362,rice
85,58,41,21.77046169,80.31964408,7.038096361,226.6555374,rice
...
```

**Requirements**:
- 2,200 rows (100 per crop class)
- 8 columns (7 features + 1 label)
- 22 crop classes
- No missing values

### Option 2: Verify Dataset

```bash
python -c "import pandas as pd; df = pd.read_csv('Crop-recommendation.csv'); print(f'✓ Dataset loaded: {len(df)} rows, {len(df.columns)} columns')"
```

Expected output:
```
✓ Dataset loaded: 2200 rows, 8 columns
```

---

## 4. Verification

### Run System Tests

```bash
python test_system.py
```

**Expected Results**:
- ✓ Dependencies (7/7 packages)
- ✓ Project Structure (all files present)
- ✓ Data Loading (dataset accessible)
- ✓ Preprocessing (pipeline working)
- ✓ Model Training (functions working)
- ✓ Inference (system ready)
- ✓ Utilities (helpers working)

### Check Project Structure

```bash
# Windows
tree /F
# macOS/Linux
tree
```

Should show:
```
crop_rec/
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── preprocessing.py
│   ├── training.py
│   └── inference.py
├── Crop-recommendation.csv
├── main.py
├── requirements.txt
└── README.md
```

---

## 5. Training Models

### Run Complete Training Pipeline

```bash
python main.py
```

**What happens**:
1. ✓ Loads and validates dataset
2. ✓ Performs data preprocessing
3. ✓ Trains 5 different models
4. ✓ Performs cross-validation
5. ✓ Optimizes hyperparameters
6. ✓ Evaluates on test set
7. ✓ Selects best model
8. ✓ Saves model and preprocessor

**Duration**: 5-10 minutes (depends on CPU)

**Progress Output**:
```
================================================================================
STEP 1: DATA LOADING AND VALIDATION
================================================================================
✓ Dataset loaded and validated successfully
  - Samples: 2,200
  - Features: 7
  - Classes: 22

================================================================================
STEP 2: DATA PREPROCESSING
================================================================================
✓ Data preprocessing completed
  - Training samples: 1,760
  - Test samples: 440
  - Features scaled: Yes

... [continues] ...

================================================================================
PIPELINE EXECUTION COMPLETED SUCCESSFULLY
================================================================================

📁 Generated Files:
  - models/best_model.pkl
  - models/preprocessor.pkl
  - models/model_metadata.json
  - reports/feature_importance.csv

📊 Key Results:
  - Best Model: XGBoost
  - Test Accuracy: 0.9954
  - CV Accuracy: 0.9945 (±0.0023)

✅ System is ready for production deployment!
```

### Verify Training Success

```bash
# Check if models were created
ls models/
# Should show: best_model.pkl, preprocessor.pkl, model_metadata.json
```

---

## 6. Making Predictions

### Option 1: Using Python Script

```bash
python example_usage.py
```

This demonstrates 7 different usage scenarios.

### Option 2: Using Python Interactive Shell

```bash
python
```

```python
from src.inference import CropRecommendationSystem

# Initialize
recommender = CropRecommendationSystem()
recommender.load_models()

# Make prediction
input_data = {
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 20.87,
    'humidity': 82.0,
    'ph': 6.5,
    'rainfall': 202.9
}

crop = recommender.predict(input_data)
print(f"Recommended Crop: {crop}")
# Output: Recommended Crop: rice

# Get confidence
crop, probs = recommender.predict(input_data, return_probabilities=True)
print(f"Confidence: {probs.max():.2%}")
# Output: Confidence: 100.00%

# Get top 3 recommendations
top_3 = recommender.get_top_n_recommendations(input_data, n=3)
for i, (crop, confidence) in enumerate(top_3, 1):
    print(f"{i}. {crop}: {confidence:.1%}")
```

### Option 3: Using Jupyter Notebook

```bash
jupyter notebook notebooks/CropRecommendation_Complete.ipynb
```

Run all cells for complete analysis.

---

## 7. Troubleshooting

### Issue 1: "ModuleNotFoundError"

**Error**: `ModuleNotFoundError: No module named 'sklearn'`

**Solution**:
```bash
pip install scikit-learn
# Or reinstall all dependencies
pip install -r requirements.txt
```

### Issue 2: "Dataset file not found"

**Error**: `FileNotFoundError: Dataset file not found: Crop-recommendation.csv`

**Solution**:
- Ensure `Crop-recommendation.csv` is in project root (`d:\crop_rec\`)
- Check filename spelling (case-sensitive on Linux/macOS)
- Verify file is not empty

```bash
# Check file exists
ls Crop-recommendation.csv

# Check file size
ls -lh Crop-recommendation.csv
```

### Issue 3: Python Version Issues

**Error**: `SyntaxError` or version-related errors

**Solution**:
```bash
# Check Python version
python --version

# If < 3.8, upgrade Python
# Windows: Download from python.org
# macOS: brew upgrade python3
# Linux: sudo apt upgrade python3
```

### Issue 4: Permission Errors

**Error**: `PermissionError: [Errno 13] Permission denied`

**Solution**:
```bash
# Windows: Run as Administrator
# macOS/Linux: Use sudo or fix permissions
chmod +x main.py
```

### Issue 5: Memory Errors

**Error**: `MemoryError` during training

**Solution**:
1. Close other applications
2. Reduce tuning iterations in `main.py`:
   ```python
   TUNING_ITERATIONS = 20  # Instead of 50
   ```
3. Use fewer models:
   ```python
   top_models = ['XGBoost', 'RandomForest']  # Only top 2
   ```

### Issue 6: Slow Training

**Problem**: Training takes > 20 minutes

**Solution**:
```python
# In main.py, set:
TUNING_ITERATIONS = 20  # Faster
CV_FOLDS = 3  # Instead of 5
n_jobs = -1  # Use all CPU cores (already default)
```

### Issue 7: Import Errors in Jupyter

**Error**: `ImportError` in notebook

**Solution**:
```bash
# Install ipykernel in virtual environment
pip install ipykernel
python -m ipykernel install --user --name=crop_rec

# Then select 'crop_rec' kernel in Jupyter
```

### Issue 8: Plotting Issues

**Error**: Figures not displaying

**Solution**:
```python
# Add to notebook
%matplotlib inline

# Or use different backend
import matplotlib
matplotlib.use('Agg')
```

### Issue 9: XGBoost Installation Issues

**Error**: XGBoost installation fails

**Solution**:
```bash
# Try specific version
pip install xgboost==1.5.0

# Or use conda
conda install -c conda-forge xgboost
```

### Issue 10: Test Failures

**Error**: `python test_system.py` shows failures

**Solution**:
1. Check each failed test
2. Install missing dependencies
3. Verify file structure
4. Check Python version

```bash
# Detailed test output
python test_system.py -v
```

---

## Quick Reference Commands

```bash
# Setup
cd d:\crop_rec
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Test
python test_system.py

# Train
python main.py

# Use
python example_usage.py

# Jupyter
jupyter notebook

# Deactivate virtual environment
deactivate
```

---

## Success Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Dataset file in place
- [ ] System tests passing
- [ ] Models trained successfully
- [ ] Can make predictions
- [ ] Example code runs
- [ ] Jupyter notebook accessible

**All checked? You're ready to go! 🚀**

---

## Need Help?

1. **Check Logs**: `crop_recommendation.log`
2. **Review Documentation**: `README.md`, `PROJECT_SUMMARY.md`
3. **Run Tests**: `python test_system.py`
4. **Check Examples**: `example_usage.py`

---

## Next Steps After Setup

1. **Explore the Jupyter Notebook**: See detailed analysis and visualizations
2. **Try Different Scenarios**: Use `example_usage.py` as reference
3. **Integrate into Your Application**: Use the inference API
4. **Customize Models**: Modify hyperparameters in `src/training.py`
5. **Add More Data**: Retrain with additional samples

---

**Setup Guide Version**: 1.0.0  
**Last Updated**: June 14, 2026  
**Estimated Setup Time**: 15-20 minutes

---

**Happy Crop Recommending! 🌾**
