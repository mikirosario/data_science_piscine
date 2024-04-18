import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

def load_and_split_data(csv_file: str, test_size: int, stratify_column: str, random_seed=None) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return train_data, val_data

def validate_and_get_cmd_line_args() -> tuple[str, str]:
    if len(sys.argv) != 3:
        print("Usage: python Tree.py <path_to_training_csv_file> <path_to_test_csv_file>")
        sys.exit(1)
    training_data_csv_path, test_data_csv_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(training_data_csv_path):
        print(f"Error: The file '{training_data_csv_path}' does not exist.")
        sys.exit(1)
    if not os.path.exists(test_data_csv_path):
        print(f"Error: The file '{test_data_csv_path}' does not exist.")
        sys.exit(1)
    return training_data_csv_path, test_data_csv_path

def load_data(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    return data

# D.TREE
def decision_tree(training_csv_path: str, test_csv_path: str):
    training_data, validation_data = load_and_split_data(training_csv_path, 0.2, stratify_column='knight', random_seed=42)
    test_data = load_data(test_csv_path)
    
    # Split features and target
    X_train = training_data.drop('knight', axis=1)
    y_train = training_data['knight']
    X_val = validation_data.drop('knight', axis=1)
    y_val = validation_data['knight']
    
    # Create and train the model
    model = DecisionTreeClassifier(random_state=69)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Evaluate the model and print f1-score
    f1 = f1_score(y_val, y_pred, average='macro')
    # Make predictions on test data
    
    test_predictions = model.predict(test_data)
    print(f"F1 Score: {f1:.2f}")
    
    # Output predictions to a file
    with open('Tree.txt', 'w') as f:
        for prediction in test_predictions:
            f.write(f'{prediction}\n')
    
    # Display and save the tree as a PNG file
    plt.figure(figsize=(20,10))
    plot_tree(model, filled=True, feature_names=X_train.columns, class_names=['Jedi', 'Sith'])
    plt.savefig('tree.png')
    print("Decision tree saved as 'tree.png'.")


def random_forest(training_csv_path: str, test_csv_path: str):
    training_data, validation_data = load_and_split_data(training_csv_path, 0.2, stratify_column='knight', random_seed=42)
    test_data = load_data(test_csv_path)

    X_train = training_data.drop('knight', axis=1)
    y_train = training_data['knight']
    X_val = validation_data.drop('knight', axis=1)
    y_val = validation_data['knight']

    model = RandomForestClassifier(n_estimators=10, random_state=69)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_val_pred, average='macro')
    print(f"F1 Score: {f1:.2f}")

    y_pred = model.predict(test_data)
    with open('Tree.txt', 'w') as f:
        for prediction in y_pred:
            f.write(f'{prediction}\n')

    # Display one of the trees in the forest
    plt.figure(figsize=(20,10))
    plot_tree(model.estimators_[0], filled=True, feature_names=X_train.columns, class_names=['Jedi', 'Sith'])
    plt.savefig('tree.png')
    print("One of the decision trees saved as 'random_forest_tree.png'.")

def main():
    try:
        training_csv_path, test_csv_path = validate_and_get_cmd_line_args()
        random_forest(training_csv_path, test_csv_path)
        # decision_tree(training_csv_path, test_csv_path)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
