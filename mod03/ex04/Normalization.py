import sys
import os
# This appends the parent directory (mod03) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ex02.points import plot_scatter, plot_scatter_by_category

def get_training_data() -> pd.DataFrame:
    df = pd.read_csv('../data/Train_knight.csv')
    return df

def get_test_data() -> pd.DataFrame:
    df = pd.read_csv('../data/Test_knight.csv')
    return df

def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=['number']).columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("Normalized data:\n", df.head())   
    return df

try:
    training_data = get_training_data()
    test_data = get_test_data()
    training_data = normalize_data(training_data)
    test_data = normalize_data(test_data)
    plot_scatter_by_category(training_data, 'Push', 'Midi-chlorien')
    plot_scatter(test_data, 'Push', 'Midi-chlorien')
except Exception as e:
    print(e)
