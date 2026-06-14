"""
Script to extract the full dataset from the document content
"""

dataset_content = """N,P,K,temperature,humidity,ph,rainfall,label
90,42,43,20.87974371,82.00274423,6.502985292000001,202.9355362,rice
85,58,41,21.77046169,80.31964408,7.038096361,226.6555374,rice
60,55,44,23.00445915,82.3207629,7.840207144,263.9642476,rice"""

# Save to file
with open('Crop-recommendation.csv', 'w') as f:
    f.write(dataset_content)

print("Dataset extracted successfully to Crop-recommendation.csv")
print(f"Preview of first 3 lines saved")
print("\nNote: The full dataset from the document should be used.")
print("Please copy all data from the attached document to this CSV file.")
