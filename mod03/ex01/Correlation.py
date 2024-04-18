import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_training_data() -> pd.DataFrame:
    df = pd.read_csv('../data/Train_knight.csv')
    return df

try:
    df_train = get_training_data()
    # Map string categories to integers
    category_mapping = {'Jedi': 1, 'Sith': 0}
    df_train['knight'] = df_train['knight'].replace(category_mapping)
    correlation_matrix = df_train.corr(method='pearson')
    target_correlation = correlation_matrix['knight']
    target_correlation = target_correlation.sort_values(ascending=False)
    print(target_correlation.to_string())
except Exception as e:
    print(e)
