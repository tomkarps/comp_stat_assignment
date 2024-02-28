# Regression Model Evaluation Project

## Project Overview

This project investigates the phenomenon where the R^2 score, commonly used to evaluate the fit of regression models, can be misleading when a large number of predictors are used relative to the number of observations. A Monte Carlo simulation was conducted using synthetic data to demonstrate the problem and explored the use of the adjusted R^2 score as a potential solution. The project was further extended to a real-world dataset — the diabetes dataset from sklearn — to apply and validate the previous findings.

## Key Findings

- Adding predictors to a regression model can inflate the R^2 score artificially.
- The adjusted R^2 score provides a more accurate measure of model performance by penalizing excessive use of predictors.
- The diabetes dataset was used to demonstrate the adjusted R^2 in practice, along with the creation of synthetic features to highlight the effects on model evaluation metrics.

## Getting Started

To run the project on your local machine, follow these steps:

1. **Clone the Repository**

   ```sh
   git clone https://github.com/tomkarps/comp_stat_assignment
   cd your-repository-name

2. **Create a virtual environment**

   python -m venv venv
   .\venv\Scripts\activate

3. **Install Dependencies**
    
   pip install -r requirements.txt


## Usage

After setting up the environment and installing the dependencies, you can run the simulation and model evaluation scripts as follows:

python simulate.py
python baseline_model.py