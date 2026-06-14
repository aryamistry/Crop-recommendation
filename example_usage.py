"""
Example usage of the Crop Recommendation System
Demonstrates various use cases and API functionality

Author: Principal ML Engineer
"""

import sys
sys.path.append('src')

from inference import CropRecommendationSystem
import pandas as pd


def example_1_single_prediction():
    """Example 1: Single crop recommendation"""
    print("=" * 80)
    print("EXAMPLE 1: Single Crop Recommendation")
    print("=" * 80)
    
    # Initialize system
    recommender = CropRecommendationSystem()
    recommender.load_models()
    
    # Define input
    input_data = {
        'N': 90,
        'P': 42,
        'K': 43,
        'temperature': 20.87,
        'humidity': 82.0,
        'ph': 6.5,
        'rainfall': 202.9
    }
    
    print("\nInput Conditions:")
    for feature, value in input_data.items():
        print(f"  {feature:15s}: {value}")
    
    # Make prediction
    prediction = recommender.predict(input_data)
    
    print(f"\n✓ Recommended Crop: {prediction}")
    print()


def example_2_with_confidence():
    """Example 2: Recommendation with confidence scores"""
    print("=" * 80)
    print("EXAMPLE 2: Recommendation with Confidence Scores")
    print("=" * 80)
    
    recommender = CropRecommendationSystem()
    recommender.load_models()
    
    input_data = {
        'N': 85,
        'P': 58,
        'K': 41,
        'temperature': 21.77,
        'humidity': 80.32,
        'ph': 7.04,
        'rainfall': 226.66
    }
    
    print("\nInput Conditions:")
    for feature, value in input_data.items():
        print(f"  {feature:15s}: {value}")
    
    # Get prediction with probabilities
    prediction, probabilities = recommender.predict(
        input_data,
        return_probabilities=True
    )
    
    confidence = probabilities.max()
    
    print(f"\n✓ Recommended Crop: {prediction}")
    print(f"  Confidence: {confidence:.2%}")
    print()


def example_3_top_recommendations():
    """Example 3: Get top N recommendations"""
    print("=" * 80)
    print("EXAMPLE 3: Top 5 Crop Recommendations")
    print("=" * 80)
    
    recommender = CropRecommendationSystem()
    recommender.load_models()
    
    input_data = {
        'N': 60,
        'P': 55,
        'K': 44,
        'temperature': 23.00,
        'humidity': 82.32,
        'ph': 7.84,
        'rainfall': 263.96
    }
    
    print("\nInput Conditions:")
    for feature, value in input_data.items():
        print(f"  {feature:15s}: {value}")
    
    # Get top 5 recommendations
    top_5 = recommender.get_top_n_recommendations(input_data, n=5)
    
    print("\nTop 5 Recommended Crops:")
    print("-" * 50)
    for i, (crop, confidence) in enumerate(top_5, 1):
        print(f"  {i}. {crop:20s} - Confidence: {confidence:.2%}")
    print()


def example_4_batch_prediction():
    """Example 4: Batch predictions for multiple fields"""
    print("=" * 80)
    print("EXAMPLE 4: Batch Predictions for Multiple Fields")
    print("=" * 80)
    
    recommender = CropRecommendationSystem()
    recommender.load_models()
    
    # Create batch data for multiple fields
    batch_data = pd.DataFrame([
        {
            'N': 90, 'P': 42, 'K': 43,
            'temperature': 20.87, 'humidity': 82.0,
            'ph': 6.5, 'rainfall': 202.9
        },
        {
            'N': 85, 'P': 58, 'K': 41,
            'temperature': 21.77, 'humidity': 80.32,
            'ph': 7.04, 'rainfall': 226.66
        },
        {
            'N': 60, 'P': 55, 'K': 44,
            'temperature': 23.00, 'humidity': 82.32,
            'ph': 7.84, 'rainfall': 263.96
        },
        {
            'N': 119, 'P': 25, 'K': 51,
            'temperature': 24.40, 'humidity': 79.19,
            'ph': 7.23, 'rainfall': 90.80
        },
        {
            'N': 22, 'P': 130, 'K': 196,
            'temperature': 22.75, 'humidity': 90.69,
            'ph': 5.52, 'rainfall': 110.43
        }
    ])
    
    print(f"\nProcessing {len(batch_data)} fields...")
    
    # Make batch predictions
    predictions = recommender.predict_batch(batch_data)
    
    # Display results
    print("\nBatch Prediction Results:")
    print("-" * 80)
    for i, (pred, row) in enumerate(zip(predictions, batch_data.to_dict('records')), 1):
        print(f"\nField {i}:")
        print(f"  Conditions: N={row['N']}, P={row['P']}, K={row['K']}, "
              f"T={row['temperature']:.1f}°C, H={row['humidity']:.1f}%")
        print(f"  → Recommended Crop: {pred}")
    print()


def example_5_input_validation():
    """Example 5: Input validation and error handling"""
    print("=" * 80)
    print("EXAMPLE 5: Input Validation and Error Handling")
    print("=" * 80)
    
    recommender = CropRecommendationSystem()
    recommender.load_models()
    
    # Test cases
    test_cases = [
        {
            'name': 'Valid Input',
            'data': {
                'N': 90, 'P': 42, 'K': 43,
                'temperature': 20.87, 'humidity': 82.0,
                'ph': 6.5, 'rainfall': 202.9
            }
        },
        {
            'name': 'Missing Feature',
            'data': {
                'N': 90, 'P': 42,
                'temperature': 20.87, 'humidity': 82.0,
                'ph': 6.5, 'rainfall': 202.9
            }
        },
        {
            'name': 'Out of Range pH',
            'data': {
                'N': 90, 'P': 42, 'K': 43,
                'temperature': 20.87, 'humidity': 82.0,
                'ph': 15.0, 'rainfall': 202.9
            }
        },
        {
            'name': 'Negative Values',
            'data': {
                'N': -10, 'P': 42, 'K': 43,
                'temperature': 20.87, 'humidity': 82.0,
                'ph': 6.5, 'rainfall': 202.9
            }
        }
    ]
    
    print("\nValidation Test Results:")
    print("-" * 80)
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        is_valid, message = recommender.validate_input(test['data'])
        
        if is_valid:
            print(f"  ✓ Valid - {message}")
            try:
                prediction = recommender.predict(test['data'])
                print(f"  Prediction: {prediction}")
            except Exception as e:
                print(f"  ✗ Prediction failed: {str(e)}")
        else:
            print(f"  ✗ Invalid - {message}")
    print()


def example_6_detailed_explanation():
    """Example 6: Get detailed prediction explanation"""
    print("=" * 80)
    print("EXAMPLE 6: Detailed Prediction Explanation")
    print("=" * 80)
    
    recommender = CropRecommendationSystem()
    recommender.load_models()
    
    input_data = {
        'N': 90,
        'P': 42,
        'K': 43,
        'temperature': 20.87,
        'humidity': 82.0,
        'ph': 6.5,
        'rainfall': 202.9
    }
    
    print("\nInput Conditions:")
    for feature, value in input_data.items():
        print(f"  {feature:15s}: {value}")
    
    # Get detailed explanation
    explanation = recommender.explain_prediction(input_data)
    
    print(f"\n✓ Recommended Crop: {explanation['recommended_crop']}")
    print(f"  Confidence: {explanation['confidence']:.2%}")
    
    print("\nTop 3 Alternatives:")
    for i, (crop, prob) in enumerate(explanation['top_3_recommendations'], 1):
        print(f"  {i}. {crop:15s}: {prob:.2%}")
    
    if explanation['feature_importance']:
        print("\nKey Influencing Features:")
        # Sort by importance
        sorted_features = sorted(
            explanation['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, importance in sorted_features[:5]:
            print(f"  {feature:15s}: {importance:.2%}")
    print()


def example_7_comparing_scenarios():
    """Example 7: Compare different scenarios"""
    print("=" * 80)
    print("EXAMPLE 7: Comparing Different Field Scenarios")
    print("=" * 80)
    
    recommender = CropRecommendationSystem()
    recommender.load_models()
    
    scenarios = {
        'High Nitrogen': {
            'N': 140, 'P': 40, 'K': 40,
            'temperature': 25, 'humidity': 80,
            'ph': 6.5, 'rainfall': 150
        },
        'High Phosphorus': {
            'N': 40, 'P': 140, 'K': 40,
            'temperature': 25, 'humidity': 80,
            'ph': 6.5, 'rainfall': 150
        },
        'High Potassium': {
            'N': 40, 'P': 40, 'K': 200,
            'temperature': 25, 'humidity': 80,
            'ph': 6.5, 'rainfall': 150
        },
        'Low Rainfall': {
            'N': 80, 'P': 40, 'K': 40,
            'temperature': 30, 'humidity': 50,
            'ph': 6.5, 'rainfall': 50
        },
        'High Rainfall': {
            'N': 80, 'P': 40, 'K': 40,
            'temperature': 25, 'humidity': 85,
            'ph': 6.5, 'rainfall': 250
        }
    }
    
    print("\nComparing Scenarios:")
    print("-" * 80)
    
    for scenario_name, conditions in scenarios.items():
        prediction = recommender.predict(conditions)
        top_3 = recommender.get_top_n_recommendations(conditions, n=3)
        
        print(f"\n{scenario_name}:")
        print(f"  Best Crop: {prediction} ({top_3[0][1]:.1%})")
        print(f"  Alternatives: {top_3[1][0]} ({top_3[1][1]:.1%}), "
              f"{top_3[2][0]} ({top_3[2][1]:.1%})")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("CROP RECOMMENDATION SYSTEM - USAGE EXAMPLES")
    print("=" * 80)
    print()
    
    examples = [
        ("Single Prediction", example_1_single_prediction),
        ("With Confidence", example_2_with_confidence),
        ("Top Recommendations", example_3_top_recommendations),
        ("Batch Prediction", example_4_batch_prediction),
        ("Input Validation", example_5_input_validation),
        ("Detailed Explanation", example_6_detailed_explanation),
        ("Scenario Comparison", example_7_comparing_scenarios)
    ]
    
    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except FileNotFoundError:
            print(f"\n❌ Error: Models not found. Please run main.py first to train models.")
            print("\nTo train models, run: python main.py")
            break
        except Exception as e:
            print(f"\n❌ Error in {name}: {str(e)}")
            continue
        
        if i < len(examples):
            input("\nPress Enter to continue to next example...")
            print("\n")
    
    print("=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print("\n📚 For more information, see:")
    print("  - README.md for complete documentation")
    print("  - notebooks/CropRecommendation_Complete.ipynb for detailed analysis")
    print("  - src/ for module-level documentation")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure you have:")
        print("  1. Trained the models (run: python main.py)")
        print("  2. Installed all dependencies (run: pip install -r requirements.txt)")
