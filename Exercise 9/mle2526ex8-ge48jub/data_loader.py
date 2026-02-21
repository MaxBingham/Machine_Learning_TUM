import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_housing_data(csv_path='housing.csv', test_size=0.2, random_state=42):
    """
    Loads housing data from a local CSV file.
    
    Expected Columns:
    longitude, latitude, housing_median_age, total_rooms, total_bedrooms, 
    population, households, median_income, median_house_value, ocean_proximity
    
    Returns:
        X_train, X_test: Feature matrices
        y_train, y_test: Continuous targets (Regression)
        y_train_cls, y_test_cls: Binary targets (Classification)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}. Please download the dataset.")

    # 1. Load Data
    df = pd.read_csv(csv_path)
    
    # 2. Preprocessing
    # Drop 'ocean_proximity' (categorical) for simplicity in distance calculations
    if 'ocean_proximity' in df.columns:
        df = df.drop('ocean_proximity', axis=1)
        
    # Handle missing values (total_bedrooms often has NaNs in this dataset)
    # We fill with the median of the column
    if df.isnull().values.any():
        df = df.fillna(df.median())

    # 3. Separate Features and Target
    # Target is 'median_house_value'
    y = df['median_house_value'].values
    X = df.drop('median_house_value', axis=1).values
    
    # 4. Create Binary Targets for Classification
    # Class 1 (Expensive): Value > Median
    # Class 0 (Affordable): Value <= Median
    threshold = np.median(y)
    y_cls = (y > threshold).astype(int)
    
    # 5. Split
    X_train, X_test, y_train, y_test, y_train_cls, y_test_cls = train_test_split(
        X, y, y_cls, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, y_train_cls, y_test_cls