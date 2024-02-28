import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets

def run_simulation(n_features: int):
    # Load dataset from sklearn
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    r2_scores = []
    adjusted_r2_scores = []
    regression = LinearRegression()
    sfs_forward = SequentialFeatureSelector(
        regression, 
        n_features_to_select=n_features-1, 
        direction="forward").fit(diabetes_X, diabetes_y)
    try:
        for feature in range(1, n_features + 1):
            diabetes_X_selected_features = diabetes_X[:, :feature]
            r2 = build_baseline_model(diabetes_X_selected_features, diabetes_y)
            adjusted_r2 = calculate_adjusted_r2(r2, diabetes_X_selected_features.shape[0], p=feature)
            r2_scores.append(r2)
            adjusted_r2_scores.append(adjusted_r2)
            print(f"R^2 Score with {feature} features: {r2}, adjusted R^2 score: {adjusted_r2}")
    except ValueError:
        print(f"The number of features is incorrect max_features of the dataset are {diabetes_X.shape[1]}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n_features + 1), r2_scores, label='R^2 Score')
    plt.plot(range(1, n_features + 1), adjusted_r2_scores, label='Adjusted R^2 Score', linestyle='--')
    plt.xlabel('Number of Features')
    plt.ylabel('Score')
    plt.title('R^2 and Adjusted R^2 Scores by Number of Features')
    plt.legend()
    plt.show()

def run_simulation_with_added_features():
    # Load dataset
    diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)

    # Create polynomial features
    degree = 2 
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(diabetes_x)
    
    r2_scores = []
    adjusted_r2_scores = []
    
    for feature in range(1, X_poly.shape[1] + 1):
            diabetes_X_selected_features = X_poly[:, :feature]
            r2 = build_baseline_model(diabetes_X_selected_features, diabetes_y)
            adjusted_r2 = calculate_adjusted_r2(r2, diabetes_X_selected_features.shape[0], p=feature)
            r2_scores.append(r2)
            adjusted_r2_scores.append(adjusted_r2)
            print(f"R^2 Score with {feature} features: {r2}, adjusted R^2 score: {adjusted_r2}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, X_poly.shape[1] + 1), r2_scores, label='R^2 Score')
    plt.plot(range(1, X_poly.shape[1] + 1), adjusted_r2_scores, label='Adjusted R^2 Score', linestyle='--')
    plt.xlabel('Number of Features')
    plt.ylabel('Score')
    plt.title('R^2 and Adjusted R^2 Scores with added Number of Features')
    plt.legend()
    plt.show()
    print("")

def build_baseline_model(X: pd.DataFrame, y: pd.Series) -> float:
    """Build and evaluate a baseline linear regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    # print(f"Baseline Model R^2: {r2}")
    return r2

def calculate_adjusted_r2(r2: float, n: int, p: int) -> float:
    """
    Calculate the adjusted R-squared value.

    Parameters:
    - r2: The R-squared value from the regression model.
    - n: The number of observations in the dataset.
    - p: The number of predictors used in the model (excluding the intercept).

    Returns:
    - The adjusted R-squared value.
    """
    adjusted_r2 = 1 - (((1 - r2) * (n - 1)) / (n - p - 1))
    return adjusted_r2

def main() -> None:
    # run_simulation(n_features=10)
    run_simulation_with_added_features()
if __name__ == "__main__":
    main()
