import sys
from matplotlib.colors import Normalize, Colormap
import matplotlib.pyplot as plt

def load_data(filepath) -> list[str]:
    """Reads a file and returns its contents as a list of strings, one per line."""
    with open(filepath, 'r') as file:
        data = file.read().splitlines()
    return data

def decompose_confusion_matrix(cm: list[list[int]]) -> tuple[int, int, int, int]:
    true_positives = cm[0][0]
    false_negatives = cm[0][1]
    false_positives = cm[1][0]
    true_negatives = cm[1][1]
    return true_positives, false_positives, true_negatives, false_negatives

def calculate_total(cm: list[list[int]]) -> tuple[int, int]:
    true_positives, false_positives, true_negatives, false_negatives = decompose_confusion_matrix(cm)
    jedi_total = true_positives + false_negatives
    sith_total = true_negatives + false_positives
    return jedi_total, sith_total

def calculate_precision(cm: list[list[int]]) -> tuple[float, float]:
    """Ratio of correctly predicted positives to total predicted positives (what proportion of
    predicted positives were correct)"""
    true_positives, false_positives, true_negatives, false_negatives = decompose_confusion_matrix(cm)
    jedi_precision = true_positives / (true_positives + false_positives) if true_positives > 0 else 0
    sith_precision = true_negatives / (true_negatives + false_negatives) if true_negatives > 0 else 0
    return jedi_precision, sith_precision

def calculate_recall(cm: list[list[int]]) -> tuple[float, float]:
    """Ratio of actual positives to correctly predicted positives (what proportion of actual
    positives were correctly predicted)"""
    true_positives, false_positives, true_negatives, false_negatives = decompose_confusion_matrix(cm)
    jedi_recall = true_positives / (true_positives + false_negatives) if true_positives > 0 else 0
    sith_recall = true_negatives / (true_negatives + false_positives) if true_negatives > 0 else 0
    return jedi_recall, sith_recall

def calculate_f1_score(cm: list[list[int]]) -> tuple[float, float]:
    """Weighted average of Precision and Recall. Therefore takes both false positives and false
    negatives into account. Drags result towards the lower value. """
    jedi_precision, sith_precision = calculate_precision(cm)
    jedi_recall, sith_recall = calculate_recall(cm)
    jedi_f1_score = 0 if jedi_precision == 0 or jedi_recall == 0 else 2 * (jedi_precision * jedi_recall)/(jedi_precision + jedi_recall)
    sith_f1_score = 0 if sith_precision == 0 or jedi_recall == 0 else 2 * (sith_precision * sith_recall)/(sith_precision + sith_recall)
    return jedi_f1_score, sith_f1_score

def calculate_accuracy(cm: list[list[int]]) -> float:
    """Overall correctness of the model. Ratio of correct predictions to total observations."""
    true_positives, false_positives, true_negatives, false_negatives = decompose_confusion_matrix(cm)
    total_accurate_predictions = true_positives + true_negatives
    total_predictions = total_accurate_predictions + false_negatives + false_positives
    return total_accurate_predictions/total_predictions

def calculate_luminance(color: list[float, float, float]) -> float:
    """Magic formula to calculate luminance of a color based on the RGB values. If it's below 0.5,
    white is the best contrast for the human eye, otherwise black.
    Uses the formula L = 0.299*R + 0.587*G + 0.114*B."""
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

def compute_confusion_matrix(predictions, truths) -> list[list[int]]:
    """Computes a confusion matrix from lists of predictions and actual truth values."""
    true_positives = sum(1 for pred, truth in zip(predictions, truths) if pred == "Jedi" and truth == "Jedi")
    false_negatives = sum(1 for pred, truth in zip(predictions, truths) if pred == "Sith" and truth == "Jedi")
    false_positives = sum(1 for pred, truth in zip(predictions, truths) if pred == "Jedi" and truth == "Sith")
    true_negatives = sum(1 for pred, truth in zip(predictions, truths) if pred == "Sith" and truth == "Sith")
    confusion_matrix = [[true_positives, false_negatives], [false_positives, true_negatives]]    
    print(confusion_matrix)
    return confusion_matrix

def plot_confusion_matrix(cm: list[list[int]], classes: list[str], title: str='Confusion Matrix', cmap: Colormap=plt.cm.viridis):
    """Plots a confusion matrix using matplotlib."""
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=45)
    # Get the color map
    norm: Normalize = plt.Normalize(vmin=min(map(min, cm)), vmax=max(map(max, cm)))
    colors = cmap(norm(cm))
    # Ejemplo:
    # cm = [[20, 40], [60, 80]] ⤵
    # vmin=20 and vmax=80       ⤵
    #               [[0.0, 0.25], [0.5, 0.75]].
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            # Calculate the luminance of the background color
            bg_color = colors[i, j][:3]  # RGB values
            luminance = calculate_luminance(bg_color)
            color = 'white' if luminance < 0.5 else 'black'
            plt.text(j, i, str(cm[i][j]), horizontalalignment="center", color=color, weight='bold')
    plt.tight_layout()
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.savefig('confusion_matrix.png', format='png')
    plt.close()

def print_metrics_table(jedi_precision, jedi_recall, jedi_f1_score, jedi_total,
                        sith_precision, sith_recall, sith_f1_score, sith_total, accuracy):
    minw = 10   # Minimum width for each column
    dec = 2     # Number of decimal places for float formatting
    header = f"{'Class':<{minw}}{'Precision':>{minw}}{'Recall':>{minw}}{'F1-Score':>{minw}}{'Total':>{minw}}"
    print(header)
    print('-' * len(header))
    print(f"{'Jedi':<{minw}}{jedi_precision:>{minw}.{dec}f}{jedi_recall:>{minw}.{dec}f}{jedi_f1_score:>{minw}.{dec}f}{jedi_total:>{minw}}")
    print(f"{'Sith':<{minw}}{sith_precision:>{minw}.{dec}f}{sith_recall:>{minw}.{dec}f}{sith_f1_score:>{minw}.{dec}f}{sith_total:>{minw}}")
    print(f"\n{'Accuracy':<{minw}}{accuracy:>{minw*3}.{dec}f}{jedi_total+sith_total:>{minw}}")



def main():
    """Main function to load data, compute the confusion matrix, and plot it."""
    try:
        if len(sys.argv) != 3:
            print("Usage: python script_name.py <predictions_path> <truth_path>")
            sys.exit(1)
        predictions_path, truth_path = sys.argv[1], sys.argv[2]
        predictions = load_data(predictions_path)
        truths = load_data(truth_path)
        cm = compute_confusion_matrix(predictions, truths)
        plot_confusion_matrix(cm, ["0", "1"])
        accuracy = calculate_accuracy(cm)
        jedi_precision, sith_precision = calculate_precision(cm)
        jedi_recall, sith_recall = calculate_recall(cm)
        jedi_f1_score, sith_f1_score = calculate_f1_score(cm)
        jedi_total, sith_total = calculate_total(cm)
        print_metrics_table(jedi_precision, jedi_recall, jedi_f1_score, jedi_total,
                    sith_precision, sith_recall, sith_f1_score, sith_total, accuracy)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
