import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_housing_data(test_size=0.2, random_state=42, file_path='housing.csv'):
    """
    Loads California housing data.
    """
    df = pd.read_csv(file_path)
    if 'ocean_proximity' in df.columns: df = df.drop('ocean_proximity', axis=1)
    if df.isnull().values.any(): df = df.fillna(df.median())
    y = df['median_house_value'].values
    X = df.drop('median_house_value', axis=1).values
    feature_names = df.drop('median_house_value', axis=1).columns.tolist()

    # Binary Classification Target (Median Split)
    threshold = np.median(y)
    y_cls = (y > threshold).astype(int)
    
    return train_test_split(X, y, y_cls, test_size=test_size, random_state=random_state), feature_names