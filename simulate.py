# third party imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import r2_score
from typing import Tuple


def generate_data(n_observations: int, n_predictors: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a synthetic dataset with a specified number of observations and predictors.
    
    Parameters:
    - n_observations (int): The number of observations in the dataset.
    - n_predictors (int): The number of predictors for each observation.
    
    Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing the predictor matrix x 
      and the target variable vector y.
    """
    x = np.random.rand(n_observations, n_predictors)
    y = np.random.rand(n_observations)
    return x, y

def fit_model(X: np.ndarray, y: np.ndarray) -> Tuple[LinearRegression, np.ndarray]:
    """
    Fits a linear regression model to the synthetic dataset and predicts the target variable.
    
    Parameters:
    - x (np.ndarray): The predictor matrix.
    - y (np.ndarray): The target variable vector.
    
    Returns:
    - Tuple[LinearRegression, np.ndarray]: A tuple containing the fitted LinearRegression 
      model and the predicted values.
    """
    model = LinearRegression()
    model_fit = model.fit(X, y)
    y_pred = model_fit.predict(X)
    if X.shape[1] != 1:
        sfs_forward = SequentialFeatureSelector(
            model, 
            n_features_to_select="auto", 
            direction="forward").fit(X, y)
        sfs_forward.get_support()
    return model, y_pred

def calculate_r2(y: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the R^2 score for the fitted model.
    
    Parameters:
    - y (np.ndarray): The actual target variable values.
    - y_pred (np.ndarray): The predicted target variable values by the model.
    
    Returns:
    - float: The R^2 score.
    """
    return r2_score(y, y_pred)

def run_simulation(n_observations: int, max_predictors: int, n_iterations: int) -> np.ndarray:
    """
    Runs the Monte Carlo simulation to illustrate the problem with R^2 in datasets with a large number of predictors.
    
    Parameters:
    - n_observations (int): The number of observations in each synthetic dataset.
    - max_predictors (int): The maximum number of predictors to test in the simulation.
    - n_iterations (int): The number of iterations for the Monte Carlo simulation.
    
    Returns:
    - np.ndarray: An array containing the mean R^2 values for each number of predictors used in the simulation.
    """
    mean_r2_values = np.zeros(max_predictors)
    
    for n_predictors in range(1, max_predictors + 1):
        r2_scores = []
        
        for _ in range(n_iterations):
            try:
                x, y = generate_data(n_observations, n_predictors)
                _, y_pred = fit_model(x, y)
                r2 = calculate_r2(y, y_pred)
                r2_scores.append(r2)
            except Exception as e:
                print(f"An error has occurred: {e}")
        mean_r2_values[n_predictors - 1] = np.mean(r2_scores)
    
    return mean_r2_values

if __name__ == "__main__":
    n_obs = 200
    max_predictors = 100
    n_iterations = 100

    results = run_simulation(n_obs, max_predictors, n_iterations)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_predictors + 1), results, marker='o', linestyle='-', color='b')
    plt.title('R^2 Value Trend with Increasing Number of Predictors')
    plt.xlabel('Number of Predictors')
    plt.ylabel('Mean R^2 Value')
    plt.grid(True)
    plt.show()
    