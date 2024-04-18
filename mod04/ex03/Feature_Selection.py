import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Load data
df = pd.read_csv("../data/Train_knight.csv")

# Selecting only the predictors ('knight' is the dependent variable)
X = df.drop('knight', axis=1)

# This function calculates VIFs and excludes 'const'
def calculate_vifs(input_data) -> pd.Series:
    X_const = add_constant(input_data)
    vifs = pd.Series(
        [variance_inflation_factor(X_const.values, i) 
         for i in range(1, X_const.shape[1])],  # Skip 'const' at index 0
        index=X_const.columns[1:]  # Skip 'const'
    )
    return vifs

# Iteratively remove features with VIF greater than 5
while True:
    vifs = calculate_vifs(X)
    max_vif = vifs.max()
    if max_vif < 5:
        break
    # Find feature with the maximum VIF and drop it
    feature_to_drop = vifs.idxmax()
    X = X.drop(columns=[feature_to_drop])
    print(f"Dropped {feature_to_drop} with VIF={max_vif}")

# Remaining features with their VIFs
print("Remaining features and their VIFs:")
print(calculate_vifs(X).to_string())
