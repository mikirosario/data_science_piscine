import sys
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def validate_and_get_cmd_line_args() -> tuple[str, str]:
    if len(sys.argv) != 3:
        print("Usage: python KNN.py <path_to_training_csv_file> <path_to_test_csv_file>")
        sys.exit(1)
    training_data_csv_path, test_data_csv_path = sys.argv[1], sys.argv[2]
    if not os.path.exists(training_data_csv_path) or not os.path.exists(test_data_csv_path):
        print(f"Error: One or more specified files do not exist.")
        sys.exit(1)
    return training_data_csv_path, test_data_csv_path

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

def load_data(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    return data

def knn(training_csv_path: str, test_csv_path: str):
    training_data, validation_data = load_and_split_data(training_csv_path, test_size=0.2, stratify_column='knight', random_seed=42)
    test_data = load_data(test_csv_path)
    
    scaler = StandardScaler()
    X_train = training_data.drop('knight', axis=1)
    y_train = training_data['knight']
    X_val = validation_data.drop('knight', axis=1)
    y_val = validation_data['knight']
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    k_values = range(1, 26)
    precisions = []
    f1_scores = []
    recalls = []

    best_k = 1
    best_precision = 0
    best_f1 = 0
    best_knn: KNeighborsClassifier = None
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_val_scaled)
        precision = precision_score(y_val, y_pred, pos_label='Jedi')
        f1 = f1_score(y_val, y_pred, pos_label='Jedi')
        recall = recall_score(y_val, y_pred, pos_label='Jedi')
        precisions.append(precision)
        f1_scores.append(f1)
        recalls.append(recall)            
        if precision > best_precision:
            best_precision = precision
            best_f1 = f1
            best_k = k
            best_knn = knn

    if best_knn is not None:
        print(f"Best k: {best_k}, Best Precision: {best_precision:.2f}, Best F1 Score: {best_f1:.2f}")
        X_test_scaled = scaler.transform(test_data)
        predictions = best_knn.predict(X_test_scaled)
        # Save the best knn predictions to a file
        with open('KNN.txt', 'w') as file:
            for prediction in predictions:
                file.write(f'{prediction}\n')

        plt.figure(figsize=(10, 5))
        plt.plot(k_values, precisions, marker='o', label='Precision')
        plt.plot(k_values, f1_scores, marker='x', label='F1')
        plt.plot(k_values, recalls, marker='^', label='Recall')
        plt.title('Precision Over Number of Neighbors')
        plt.xlabel('k values (Number of Neighbours)')
        plt.ylabel('precision')
        plt.grid(True)
        plt.legend()
        plt.savefig('knn_precision.png')

def knn_cross_val(training_csv_path: str, test_csv_path: str):
    training_data, validation_data = load_and_split_data(training_csv_path, 0.2, stratify_column='knight')
    test_data = load_data(test_csv_path)
    
    scaler = StandardScaler()
    X_train = training_data.drop('knight', axis=1)
    y_train = training_data['knight']
    X_val = validation_data.drop('knight', axis=1)
    y_val = validation_data['knight']
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_val)

    k_values = range(1, 26)
    average_precisions = []
    average_f1_scores = []
    average_recalls = []

    best_k = 1
    best_precision = 0
    best_f1 = 0

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    precision_scorer = make_scorer(precision_score, pos_label='Jedi')
    f1_scorer = make_scorer(f1_score, pos_label='Jedi')
    recall_scorer = make_scorer(recall_score, pos_label='Jedi')


    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        precisions = cross_val_score(knn, X_train_scaled, y_train, cv=skf, scoring=precision_scorer)
        f1s = cross_val_score(knn, X_train_scaled, y_train, cv=skf, scoring=f1_scorer)
        recalls = cross_val_score(knn, X_train_scaled, y_train, cv=skf, scoring=recall_scorer)
        mean_precision = np.mean(precisions)
        mean_f1 = np.mean(f1s)
        mean_recall = np.mean(recalls)
        average_precisions.append(mean_precision)
        average_f1_scores.append(mean_f1)
        average_recalls.append(mean_recall)
        if mean_precision > best_precision:
            best_precision = mean_precision
            best_f1 = mean_f1
            best_k = k

    # Re-fit the model with the best k value and predict
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train_scaled, y_train)
    test_data_scaled = scaler.transform(test_data)
    best_knn_predictions = best_knn.predict(test_data_scaled)
    # Save the predictions to a file
    with open('knn_cross_val_predictions.txt', 'w') as file:
        for prediction in best_knn_predictions:
            file.write(f'{prediction}\n')

    print(f"Best k: {best_k}, Best Precision: {best_precision:.2f}, Best F1 Score: {best_f1:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(k_values, average_precisions, marker='o', label='Average Precision')
    plt.plot(k_values, average_f1_scores, marker='x', label='Average F1 Score')
    plt.plot(k_values, average_recalls, marker='^', label='Average Recall')
    plt.title('KNN Performance Evaluation with Cross-Validation')
    plt.xlabel('k values (Number of Neighbours)')
    plt.ylabel('Metric Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('knn_precision.png')

def main():
    try:
        training_csv_path, test_csv_path = validate_and_get_cmd_line_args()
        knn(training_csv_path, test_csv_path)
        # knn_cross_val(training_csv_path, test_csv_path)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
