import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_test_data() -> pd.DataFrame:
    df_test = pd.read_csv('../data/Test_knight.csv')
    return df_test


def get_training_data() -> pd.DataFrame:
    df_test = pd.read_csv('../data/Train_knight.csv')
    return df_test


def get_subplot_row_count(subplots_per_row: int, total_subplots: int) -> int:
    """
    Calculates the number of rows required for a given number of subplots distributed in specified
    columns per row.    
    Parameters:
        subplots_per_row (int): The number of subplots in each row.
        total_subplots (int): The total number of subplots to display.    
    Returns:
        int: The number of rows needed to accommodate the subplots.
    """
    return math.ceil(total_subplots / subplots_per_row)


def get_histogram_info(data: pd.DataFrame, subplots_per_row: int) -> tuple[plt.Axes, pd.DataFrame]:
    """
    Prepares and returns information necessary for histogram plotting, including the figure, axes,
    and numeric columns.
    Parameters:
        data (pd.DataFrame): The dataset to analyze.
        subplots_per_row (int): The number of subplots in each row.    
    Returns:
        tuple[plt.Axes, pd.DataFrame]: A tuple containing the matplotlib figure, axes, and array of
        numeric column names.
    """
    # Number of columns with numeric data
    numeric_cols = data.select_dtypes(include=['number']).columns
    numeric_cols_count = len(numeric_cols)
    # Calculate the number of rows needed, with up to subplots_per_row histograms per row
    subplot_row_count = get_subplot_row_count(subplots_per_row, numeric_cols_count)
        # Set up the matplotlib figure and axes
    fig, axes = plt.subplots(nrows=subplot_row_count,
                             ncols=subplots_per_row,
                             figsize=(25, 4 * subplot_row_count)) # width and height in inches
    axes: np.ndarray[plt.Axes] = axes.flatten() # Make the subplot array one dimension to iterate in single loop
    return fig, axes, numeric_cols


def create_histograms(data: pd.DataFrame,
                      subplots_color: str,
                      alpha_channel: float,
                      numeric_cols: np.ndarray[str],
                      bins_per_subplot: int,
                      axes: np.ndarray[plt.Axes]):
    """
    Populates a set of axes with histograms for each specified numeric column in the data.    
    Parameters:
        data (pd.DataFrame): The dataset containing the data to plot.
        subplots_color (str): Color of the histograms.
        alpha_channel (float): Transparency level of the histogram bars.
        numeric_cols (np.ndarray[str]): Array of column names to plot histograms for.
        bins_per_subplot (int): Number of bins for each histogram.
        axes (np.ndarray[plt.Axes]): Array of matplotlib axes objects to plot the histograms on.
    """
    for i, col in enumerate(numeric_cols):
        axes[i].hist(data[col].dropna(), bins=bins_per_subplot, color=subplots_color, alpha=alpha_channel)
        axes[i].set_title(col)
        axes[i].set_ylabel('Frequency')
    # Hide any unused subplot slots if the number of columns is not a perfect multiple of subplots_per_row
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')


def create_histograms_by_category(data: pd.DataFrame,
                      target_column: str,
                      target_category_colors: dict[str, str],
                      alpha_channel: float,
                      numeric_cols: np.ndarray[str],
                      bins_per_subplot: int,
                      axes: np.ndarray[plt.Axes]):
    """
    Populates a set of axes with histograms, each colored by categories from a specific column in
    the dataset.
    Parameters:
        data (pd.DataFrame): The dataset containing the data to plot.
        target_column (str): The column in the dataset to use for categorization.
        target_category_colors (dict[str, str]): A dictionary mapping categories to colors.
        alpha_channel (float): Transparency level of the histogram bars.
        numeric_cols (np.ndarray[str]): Array of column names to plot histograms for.
        bins_per_subplot (int): Number of bins for each histogram.
        axes (np.ndarray[plt.Axes]): Array of matplotlib axes objects to plot the histograms on.
    """
    categories = data[target_column].unique() 
    for i, col in enumerate(numeric_cols):
        for category in categories:
            subset = data[data[target_column] == category]
            axes[i].hist(subset[col].dropna(),
                         bins=bins_per_subplot,
                         alpha=alpha_channel,
                         color=target_category_colors[category],
                         label=category)
        axes[i].set_title(col)
    # Hide any unused subplot slots if the number of columns is not a perfect multiple of subplots_per_row
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')


def generate_histograms(data: pd.DataFrame,
                    bins_per_subplot: int,
                    subplots_per_row: int,
                    subplots_color: str,
                    alpha_channel: float):
    """
    Generates and saves histograms for all numeric columns in the data to a PNG file.
    Parameters:
        data (pd.DataFrame): The dataset to plot.
        bins_per_subplot (int): Number of bins for each histogram.
        subplots_per_row (int): The number of subplots in each row.
        subplots_color (str): Color of the histograms.
        alpha_channel (float): Transparency level of the histogram bars.
    """
    fig, axes, numeric_cols = get_histogram_info(data, subplots_per_row)
    create_histograms(data=data,
                      subplots_color=subplots_color,
                      alpha_channel=alpha_channel,
                      numeric_cols=numeric_cols,
                      bins_per_subplot=bins_per_subplot,
                      axes=axes)
    plt.tight_layout()
    plt.savefig('subplots.png', format='png')
    plt.close(fig)


def generate_histograms_by_category(data: pd.DataFrame,
                                    target_column: str,
                                    target_category_colors: dict[str, str],
                                    bins_per_subplot: int,
                                    subplots_per_row: int,
                                    alpha_channel: float):
    """
    Generates and saves categorized histograms for numeric columns in the data to a PNG file,
    coloring them based on specified categories.
    Parameters:
        data (pd.DataFrame): The dataset to plot.
        target_column (str): The column to use for categorizing the data.
        target_category_colors (dict[str, str]): Colors associated with each category.
        bins_per_subplot (int): Number of bins for each histogram.
        subplots_per_row (int): The number of subplots in each row.
        alpha_channel (float): Transparency level of the histogram bars.
    """
    fig, axes, numeric_cols = get_histogram_info(data, subplots_per_row)
    create_histograms_by_category(data=data,
                                  target_column=target_column,
                                  target_category_colors=target_category_colors,
                                  alpha_channel=alpha_channel,
                                  numeric_cols=numeric_cols,
                                  bins_per_subplot=bins_per_subplot,
                                  axes=axes)
    plt.tight_layout()
    plt.savefig('subplots_target.png', format='png')
    plt.close(fig)


try:
    df_test: pd.DataFrame = get_test_data()
    generate_histograms(data=df_test,
                        bins_per_subplot=40,
                        subplots_per_row=5,
                        subplots_color='green',
                        alpha_channel=0.7)
    df_train: pd.DataFrame = get_training_data()
    generate_histograms_by_category(data=df_train,
                                    target_column='knight',
                                    target_category_colors={'Jedi': 'blue', 'Sith': 'red'},
                                    bins_per_subplot=40,
                                    subplots_per_row=5,
                                    alpha_channel=0.5)
except Exception as e:
    print(e)
