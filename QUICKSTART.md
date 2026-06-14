# Quick Start Guide
## Crop Recommendation System

Get started in 5 minutes! ⚡

---

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 2GB free disk space

---

## Installation

```bash
# 1. Install dependencies (2-3 minutes)
pip install -r requirements.txt

# 2. Verify installation
python -c "import sklearn, xgboost; print('✓ Ready!')"
```

---

## Dataset Setup

**Option 1: Use the provided dataset**

The dataset from your document needs to be saved as `Crop-recommendation.csv` in the project root.

The file should have this structure:
```csv
N,P,K,temperature,humidity,ph,rainfall,label
90,42,43,20.87974371,82.00274423,6.502985292000001,202.9355362,rice
...
```

**Option 2: Quick test with sample data**

For testing, you can run with a small subset (though results won't be optimal).

---

## 3-Step Usage

### Step 1: Train Models (5-10 minutes)

```bash
python main.py
```

This will:
- ✅ Load and validate data
- ✅ Train 5 different models
- ✅ Perform cross-validation
- ✅ Select best model
- ✅ Save everything for production

**Expected Output:**
```
Best Model: XGBoost
Test Accuracy: 0.9954
✅ System is ready for production deployment!
```

### Step 2: Make Predictions

```python
from src.inference import CropRecommendationSystem

# Initialize
recommender = CropRecommendationSystem()
recommender.load_models()

# Predict
input_data = {
    'N': 90,           # Nitrogen
    'P': 42,           # Phosphorus
    'K': 43,           # Potassium
    'temperature': 20.87,  # °C
    'humidity': 82.0,      # %
    'ph': 6.5,
    'rainfall': 202.9      # mm
}

crop = recommender.predict(input_data)
print(f"Recommended Crop: {crop}")
# Output: Recommended Crop: rice
```

### Step 3: Explore Examples

```bash
python example_usage.py
```

See 7 different usage scenarios with explanations.

---

## Common Use Cases

### Get Top 3 Recommendations

```python
top_3 = recommender.get_top_n_recommendations(input_data, n=3)

for crop, confidence in top_3:
    print(f"{crop}: {confidence:.1%}")
```

### Batch Predictions

```python
import pandas as pd

# Multiple fields
fields = pd.DataFrame([
    {'N': 90, 'P': 42, 'K': 43, ...},
    {'N': 85, 'P': 58, 'K': 41, ...},
    ...
])

predictions = recommender.predict_batch(fields)
```

### Validate Input

```python
is_valid, message = recommender.validate_input(input_data)

if is_valid:
    prediction = recommender.predict(input_data)
else:
    print(f"Invalid input: {message}")
```

---

## Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook notebooks/CropRecommendation_Complete.ipynb
```

The notebook includes:
- 📊 Complete EDA with visualizations
- 🔍 Correlation analysis
- 🎯 Model comparison
- 📈 Performance metrics
- 🧪 Feature importance

---

## File Structure

After training, you'll have:

```
crop_rec/
├── models/
│   ├── best_model.pkl          ← Trained model
│   ├── preprocessor.pkl         ← Data transformer
│   └── model_metadata.json      ← Model info
├── reports/
│   ├── feature_importance.csv
│   └── figures/                 ← Visualizations
└── crop_recommendation.log      ← Execution log
```

---

## Troubleshooting

### Issue: "Dataset file not found"
**Solution:** Ensure `Crop-recommendation.csv` is in the project root directory.

### Issue: "ModuleNotFoundError"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "Memory Error"
**Solution:** Reduce batch size or use fewer tuning iterations in `main.py`:
```python
TUNING_ITERATIONS = 20  # Instead of 50
```

### Issue: Low accuracy
**Solution:** Ensure you're using the complete dataset (2200 samples, not a subset).

---

## Next Steps

1. **Review Results:**
   ```bash
   cat models/model_metadata.json
   ```

2. **Check Visualizations:**
   ```bash
   # Windows:
   start reports/figures/
   # Linux/Mac:
   open reports/figures/
   ```

3. **Read Full Documentation:**
   - `README.md` - Complete documentation
   - `src/` - Module documentation
   - Jupyter notebook - Interactive analysis

4. **Customize:**
   - Modify hyperparameters in `src/training.py`
   - Add new models
   - Create custom preprocessing steps

---

## Performance Expectations

With the full dataset:

| Metric | Expected Value |
|--------|---------------|
| Training Time | 5-10 minutes |
| Test Accuracy | >99% |
| Inference Time | <1ms per prediction |
| Model Size | ~50MB |

---

## API Quick Reference

```python
# Initialize
recommender = CropRecommendationSystem()
recommender.load_models()

# Single prediction
crop = recommender.predict(input_dict)

# With confidence
crop, probs = recommender.predict(input_dict, return_probabilities=True)

# Top N recommendations
top_n = recommender.get_top_n_recommendations(input_dict, n=5)

# Batch predictions
crops = recommender.predict_batch(dataframe)

# Detailed explanation
explanation = recommender.explain_prediction(input_dict)

# Input validation
is_valid, msg = recommender.validate_input(input_dict)
```

---

## Support

- **Issues:** Check `crop_recommendation.log`
- **Questions:** See `README.md`
- **Examples:** Run `python example_usage.py`

---

## Success Checklist

- [x] Dependencies installed
- [x] Dataset in place
- [x] Models trained (run `main.py`)
- [x] Predictions working
- [x] Examples explored

**You're all set! 🎉**

Start building your agricultural AI solution!
