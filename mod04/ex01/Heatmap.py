import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_training_data() -> pd.DataFrame:
    df = pd.read_csv('../data/Train_knight.csv')
    return df

try:
    df_train: pd.DataFrame = get_training_data()
    # Map string categories to integers
    category_mapping: dict[str, int] = {'Jedi': 1, 'Sith': 0}
    pd.set_option('future.no_silent_downcasting', True)
    df_train['knight'] = df_train['knight'].replace(category_mapping).infer_objects().convert_dtypes()
    # Compute the correlation matrix
    correlation_matrix = df_train.corr(method='pearson')
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm_r', center=0, vmin=-1, vmax=1, cbar=True) # gist_heat
    plt.title('Correlation Matrix Heatmap')
    plt.savefig('heatmap.png', format='png')
    plt.close()

except Exception as e:
    print(e)
