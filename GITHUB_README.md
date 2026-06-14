# 🌾 Crop Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.32%25-brightgreen.svg)](TESTING_SUMMARY.md)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)](DELIVERABLES.md)

> An intelligent AI system that recommends the most suitable crop based on soil nutrients (NPK), environmental conditions (temperature, humidity, rainfall), and soil pH. Built with production-grade code quality and comprehensive testing.

## 🎯 Key Features

- **🎓 99.32% Test Accuracy** - State-of-the-art machine learning performance
- **⚡ 3,891 predictions/second** - High-throughput batch processing
- **💾 2.64 MB Memory** - Lightweight and efficient
- **📊 22 Crop Classes** - Comprehensive crop recommendations
- **🔍 Explainable AI** - Feature importance and confidence scores
- **🚀 Production Ready** - Complete with testing, documentation, and deployment guides

## 🌟 Highlights

```python
# Simple API - Get crop recommendation in 3 lines
from src.inference import CropRecommendationSystem
recommender = CropRecommendationSystem()
recommender.load_models()

prediction = recommender.predict({
    'N': 90, 'P': 42, 'K': 43,
    'temperature': 20.87, 'humidity': 82.0,
    'ph': 6.5, 'rainfall': 202.9
})
# Output: 'rice' (98.95% confidence)
```

## 📊 Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Test Accuracy** | 99.32% | ✅ Excellent |
| **Test Precision** | 99.35% | ✅ Excellent |
| **Test Recall** | 99.32% | ✅ Excellent |
| **F1 Score** | 99.32% | ✅ Excellent |
| **Inference Speed** | 0.26 ms/sample (batch) | ✅ Very Fast |
| **Memory Usage** | 2.64 MB | ✅ Lightweight |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/aryamistry/Crop-recommendation.git
cd Crop-recommendation

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_system.py
```

### Usage

#### Option 1: Single Prediction
```python
from src.inference import CropRecommendationSystem

recommender = CropRecommendationSystem()
recommender.load_models()

# Make a prediction
result = recommender.predict({
    'N': 90, 'P': 42, 'K': 43,
    'temperature': 20.87,
    'humidity': 82.0,
    'ph': 6.5,
    'rainfall': 202.9
})
print(f"Recommended Crop: {result}")
```

#### Option 2: Batch Processing
```python
import pandas as pd

# Load your data
farms = pd.read_csv('your_farms.csv')

# Get recommendations for all farms
predictions = recommender.predict_batch(farms)
```

#### Option 3: Top-N Recommendations
```python
# Get top 3 crop recommendations with confidence scores
top_3 = recommender.get_top_n_recommendations(input_data, n=3)

for crop, confidence in top_3:
    print(f"{crop}: {confidence*100:.2f}%")
```

## 📁 Project Structure

```
crop_rec/
├── src/                          # Source code
│   ├── utils.py                  # Utility functions
│   ├── preprocessing.py          # Data preprocessing
│   ├── training.py               # Model training
│   └── inference.py              # Production inference
├── models/                       # Trained models
│   ├── best_model.pkl            # RandomForest model (99.32% accuracy)
│   ├── preprocessor.pkl          # Data transformer
│   └── model_metadata.json       # Model information
├── notebooks/                    # Jupyter notebooks
│   ├── CropRecommendation_Complete.ipynb
│   └── CropRecommendation_Part2.ipynb
├── reports/                      # Analysis reports
│   └── feature_importance.csv
├── docs/                         # Documentation
│   ├── README.md                 # Main documentation
│   ├── QUICKSTART.md            # 5-minute setup
│   ├── EFFICIENCY_TEST_RESULTS.md
│   └── TESTING_SUMMARY.md
├── main.py                       # Training pipeline
├── example_usage.py             # Usage examples
├── test_system.py               # System tests
├── test_efficiency.py           # Efficiency tests
└── requirements.txt             # Dependencies
```

## 🎯 Supported Crops (22 Classes)

| Category | Crops |
|----------|-------|
| **Cereals** | rice, maize |
| **Pulses** | chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil |
| **Fruits** | pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya |
| **Cash Crops** | cotton, jute, coffee, coconut |

## 🔬 Model Details

### Training Details
- **Dataset**: 2,200 samples (100 per crop)
- **Features**: 7 (N, P, K, temperature, humidity, pH, rainfall)
- **Algorithm**: RandomForest Classifier (Optimized)
- **Validation**: 5-Fold Stratified Cross-Validation
- **Optimization**: RandomizedSearchCV (50 iterations)

### Feature Importance
1. **Rainfall** - 23.49%
2. **Humidity** - 22.22%
3. **Potassium (K)** - 17.15%
4. **Phosphorus (P)** - 15.34%
5. **Nitrogen (N)** - 10.13%
6. **Temperature** - 6.81%
7. **pH** - 4.86%

## 📈 Benchmark Results

### Speed Performance
| Batch Size | Time per Sample | Throughput |
|------------|----------------|------------|
| 1 (single) | 94.33 ms | 10.6 pred/sec |
| 10 | 7.75 ms | 129 pred/sec |
| 100 | 1.06 ms | 946 pred/sec |
| **1000** | **0.26 ms** | **3,891 pred/sec** ⭐ |

### Deployment Options
- ✅ **Mobile App** - Single prediction mode (94ms)
- ✅ **Web API** - Batch processing (3,891 samples/sec)
- ✅ **Edge Device** - Low memory (2.64 MB)
- ✅ **Cloud Service** - Scalable batch processing

## 🧪 Testing

### Run System Tests
```bash
python test_system.py
# Output: 7/7 tests passed ✅
```

### Run Efficiency Tests
```bash
python test_efficiency.py
# Tests with random new data
# Output: 6/6 tests passed ✅
```

### Run Training Pipeline
```bash
python main.py
# Trains all models with cross-validation
# Time: ~3-5 minutes
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Complete project documentation |
| [QUICKSTART.md](QUICKSTART.md) | 5-minute quick start guide |
| [SETUP_INSTRUCTIONS.md](SETUP_INSTRUCTIONS.md) | Detailed setup guide |
| [EFFICIENCY_TEST_RESULTS.md](EFFICIENCY_TEST_RESULTS.md) | Performance benchmarks |
| [TESTING_SUMMARY.md](TESTING_SUMMARY.md) | Complete testing summary |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Technical & executive summary |

## 🎓 Academic Highlights

This project demonstrates:
- ✅ **Production-Grade Code** - Type hints, documentation, error handling
- ✅ **Rigorous Methodology** - Cross-validation, hyperparameter tuning
- ✅ **Comprehensive Testing** - System tests, efficiency tests (670+ predictions)
- ✅ **Feature Engineering** - Feature importance analysis
- ✅ **Model Comparison** - 5 models compared (RF, XGBoost, ET, GB, LR)
- ✅ **Explainability** - Confidence scores, top-N recommendations
- ✅ **Scalability** - Batch processing, low memory footprint

## 🛠️ Technology Stack

- **Language**: Python 3.8+
- **ML Framework**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Notebook**: Jupyter
- **Model Persistence**: Joblib

## 📊 Results Visualization

### Model Comparison
```
ExtraTrees       : ████████████████████ 99.38%
RandomForest     : ████████████████████ 99.32%
XGBoost          : ███████████████████▌ 99.15%
GradientBoosting : ██████████████████▌  98.69%
LogisticRegression: ████████████████▌    96.82%
```

### Feature Importance
```
Rainfall    : ███████████████████████ 23.49%
Humidity    : ██████████████████████  22.22%
Potassium   : █████████████████       17.15%
Phosphorus  : ███████████████         15.34%
Nitrogen    : ██████████              10.13%
Temperature : ██████                   6.81%
pH          : ████                     4.86%
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: Agricultural research community
- Libraries: Scikit-learn, XGBoost, Pandas, NumPy
- Inspiration: Precision agriculture and AI for sustainable farming

## 📞 Contact

- **Author**: Arya Mistry
- **GitHub**: [@aryamistry](https://github.com/aryamistry)
- **Repository**: [Crop-recommendation](https://github.com/aryamistry/Crop-recommendation)

## ⭐ Star History

If you find this project helpful, please consider giving it a star ⭐

---

**Status**: ✅ **PRODUCTION READY**  
**Version**: 1.0.0  
**Last Updated**: June 2026

Made with ❤️ for sustainable agriculture
