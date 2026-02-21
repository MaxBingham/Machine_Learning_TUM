import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
training_data = np.load('training_dataset_1.npy')
test_data = np.load('test_dataset_1.npy')

print("=== DATASET 1 ANALYSIS ===")
print(f"Training data shape: {training_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Extract X and y (scaled by 1000 as in your code)
X_train_raw = training_data[0, :]
y_train_raw = training_data[1, :]
X_train = X_train_raw / 1000
y_train = y_train_raw / 1000

X_test_raw = test_data[0, :]
y_test_raw = test_data[1, :]
X_test = X_test_raw / 1000
y_test = y_test_raw / 1000

print("\n=== RAW VALUES (before scaling) ===")
print(f"Training X range: {np.min(X_train_raw):.2f} to {np.max(X_train_raw):.2f}")
print(f"Training y range: {np.min(y_train_raw):.2f} to {np.max(y_train_raw):.2f}")
print(f"First 5 raw X values: {X_train_raw[:5]}")
print(f"First 5 raw y values: {y_train_raw[:5]}")

print("\n=== SCALED VALUES (after /1000) ===")
print(f"Training X range: {np.min(X_train):.3f} to {np.max(X_train):.3f}")
print(f"Training y range: {np.min(y_train):.3f} to {np.max(y_train):.3f}")
print(f"First 5 scaled X values: {X_train[:5]}")
print(f"First 5 scaled y values: {y_train[:5]}")

print(f"\nNumber of training points: {len(X_train)}")
print(f"Number of test points: {len(X_test)}")

print("\n=== INTERPRETATION ===")
print("Each point represents:")
print("- X coordinate: Input feature (independent variable)")
print("- y coordinate: Target output (dependent variable)")
print("- The goal is to find a function that maps X → y")
print("- Training points are used to learn the relationship")
print("- Test points are used to evaluate how well we learned")