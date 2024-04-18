import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

def validate_and_get_cmd_line_args() -> tuple[str, str]:
    if len(sys.argv) != 3:
        print("Usage: python KNN.py <path_to_training_csv_file> <path_to_test_csv_file>")
        sys.exit(1)
    training_data_csv_path, test_data_csv_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(training_data_csv_path) or not os.path.exists(test_data_csv_path):
        print(f"Error: One or more specified files do not exist.")
        sys.exit(1)
    return training_data_csv_path, test_data_csv_path

def load_and_split_data(csv_file: str, test_size: float, stratify_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    train_data, val_data = train_test_split(data, test_size=test_size, stratify=data[stratify_column], random_state=42)
    return train_data, val_data

def main():
    training_csv_path, test_csv_path = validate_and_get_cmd_line_args()
    train_data, validation_data = load_and_split_data(training_csv_path, test_size=0.2, stratify_column='knight')
    test_data = pd.read_csv(test_csv_path)
    
    scaler = StandardScaler()
    X_train = train_data.drop('knight', axis=1)
    y_train = train_data['knight']
    X_val = validation_data.drop('knight', axis=1)
    y_val = validation_data['knight']
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    clf1 = DecisionTreeClassifier(random_state=42)
    clf2 = KNeighborsClassifier(n_neighbors=11)
    clf3 = SVC(kernel='linear', probability=True, random_state=42)
    
    eclf = VotingClassifier(
        estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],
        voting='hard')

    eclf.fit(X_train_scaled, y_train)
    
    y_pred = eclf.predict(X_val_scaled)
    
    f1 = f1_score(y_val, y_pred, pos_label='Jedi')
    print(f"F1 Score on validation set: {f1:.2f}")
    
    # Run model on test data
    X_test_scaled = scaler.transform(test_data)
    y_test_pred = eclf.predict(X_test_scaled)
    
    # Write predictions for test data to a file
    with open('Voting.txt', 'w') as file:
        for prediction in y_test_pred:
            file.write(f'{prediction}\n')

if __name__ == "__main__":
    main()
