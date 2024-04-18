import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(csv_file: str, test_size: int, stratify_column: str, random_seed=None):
    """
    Load data from a CSV file, split it into training and validation sets based on a specified
    stratification column, and save them as new CSV files. This ensures that the class proportions
    are similar in both training and validation sets.

    Args:
    csv_file (str): Path to the source CSV file.
    test_size (float): Proportion of the dataset to include in the validation split.
    stratify_column (str): The column on which to stratify the data, ensuring class proportionality.
    random_seed (int, optional): The seed used by the random number generator for reproducibility. If None, the random state is unpredictable.

    Outputs:
    Two CSV files are generated:
        - 'Training_knight.csv' containing the training set.
        - 'Validation_knight.csv' containing the validation set.
    Each file omits row indices from the output CSV.
    """
    data = pd.read_csv(csv_file)
    # Split the data into training and validation sets
    train_data: pd.DataFrame
    val_data: pd.DataFrame
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=random_seed, stratify=data[stratify_column])
    train_data.to_csv('Training_knight.csv', index=False) # index=False to omit Pandas row indices
    val_data.to_csv('Validation_knight.csv', index=False)

def main():
    try:
        if len(sys.argv) != 2:
            print("Usage: python split.py <path_to_csv_file>")
            return
        elif not os.path.exists(sys.argv[1]):
            print(f"Error: The file '{sys.argv[1]}' does not exist.")
            return
        load_and_split_data(csv_file=sys.argv[1], test_size=0.2, stratify_column='knight')
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
