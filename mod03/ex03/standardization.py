import sys
import os
# This appends the parent directory (mod03) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ex02.points import plot_scatter, plot_scatter_by_category


def get_training_data() -> pd.DataFrame:
    df = pd.read_csv('../data/Train_knight.csv')
    return df

def get_test_data() -> pd.DataFrame:
    df = pd.read_csv('../data/Test_knight.csv')
    return df

def standardize_data(df: pd.DataFrame) -> pd.DataFrame:
    # Identify numeric columns (excluding any categorical columns like 'knight')
    numeric_cols = df.select_dtypes(include=['number']).columns
    scaler = StandardScaler()    
    # Standardize only the numeric columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])    
    # Print the standardized data
    print("Standardized data:\n", df.head())    
    return df

try:
    training_data = get_training_data()
    test_data = get_test_data()
    training_data = standardize_data(training_data)
    test_data = standardize_data(test_data)
    plot_scatter_by_category(training_data, 'Empowered', 'Stims')
    plot_scatter(test_data, 'Empowered', 'Stims')
except Exception as e:
    print(e)
