import pandas as pd
import matplotlib.pyplot as plt

def get_training_data() -> pd.DataFrame:
    df = pd.read_csv('../data/Train_knight.csv')
    return df

def get_test_data() -> pd.DataFrame:
    df = pd.read_csv('../data/Test_knight.csv')
    return df

def plot_scatter_by_category(data: pd.DataFrame, subset1: str, subset2: str):
    plt.figure(figsize=(10, 6))
    # Plot points for Jedi and Sith with labels for legend
    for category, color in zip(['Jedi', 'Sith'], ['blue', 'red']):
        subset = data[data['knight'] == category]
        plt.scatter(subset[f'{subset1}'], subset[f'{subset2}'], c=color, alpha=0.5, label=category)
    plt.title(f'Scatter Plot of {subset1} vs {subset2} by Knight Type in Training Data')
    plt.xlabel(f'{subset1}')
    plt.ylabel(f'{subset2}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'train_{subset1.lower()}_vs_{subset2.lower()}.png', format='png')

def plot_scatter(data: pd.DataFrame, subset1: str, subset2: str):
    plt.figure(figsize=(10, 6))
    # Plot scatter and capture the return value for legend handling
    scatter = plt.scatter(data[f'{subset1}'], data[f'{subset2}'], c='green', alpha=0.5, label='Knight')
    plt.title(f'Scatter Plot of {subset1} vs {subset2} in Test Data')
    plt.xlabel(f'{subset1}')
    plt.ylabel(f'{subset2}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'test_{subset1.lower()}_vs_{subset2.lower()}.png', format='png')

if __name__ == "__main__":
    try:
        training_data = get_training_data()
        test_data = get_test_data()
        plot_scatter_by_category(training_data, 'Empowered', 'Stims')
        plot_scatter_by_category(training_data, 'Push', 'Midi-chlorien')
        plot_scatter(test_data, 'Empowered', 'Stims')
        plot_scatter(test_data, 'Push', 'Midi-chlorien')
    except Exception as e:
        print(e)
