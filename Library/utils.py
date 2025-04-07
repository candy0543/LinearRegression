import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from Library.SubModel1 import SubModel1
from Library.SubModel2 import LinearRegressionCustomModel2
from Library.PolynomialModel import (
    PolynomialDegree5RegressionCustomModel,
)
from Library.QuadraticModel import QuadraticRegressionCustomModel


# Function to compute the optimal train size
def computeOptimalTrainSize(r):
    # Compute optimal train size
    optimalTestSize = 1 / (np.sqrt(r) + 1)
    optimalTrainSize = 1 - optimalTestSize

    return optimalTrainSize


# Function to compute the optimal parameters depending on the model type
def computeOptimalParameterss(x_train, x_test, y_train, y_test, model_type):
    # Set hyperparameters
    iterations_numbers = [
        200,
        300,
        400,
        500,
        1000,
        2000,
        3000,
        5000,
        7000,
        10000,
        20000,
        50000,
    ]  # Number of iterations
    learning_rates = [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]  # Learning rate

    # Initialise an empty array to store the results
    results = []

    # Set seed for reproducibility
    np.random.seed(0)
   
    # Computes linear regression for each hyperparameters
    for iterations_nb in iterations_numbers:
        for learning_rate in learning_rates:
            # Initialise model
            if model_type == "linear_2":
                # Linear Model 2
                model = LinearRegressionCustomModel2(learning_rate, iterations_nb)
            elif model_type == "quadratic":
                # Quadratic model
                model = QuadraticRegressionCustomModel(learning_rate, iterations_nb)
            elif model_type == "degree_5":
                # Polynomial degree 5 model
                model = PolynomialDegree5RegressionCustomModel(
                    learning_rate, iterations_nb
                )
            else:
                # Model 1
                model = SubModel1(learning_rate, iterations_nb)
            # Fit model to training data
            theta, cost = model.fit(x_train, y_train)
            # Skip if training was not successful
            if theta is None:
                continue
            # Compute metrics
            rSquare, meanAbErr, meanSqErr, rootMeanSqErr = model.metrics(x_test, y_test)
            # Save results
            results.append(
                {
                    "iterations_nb": iterations_nb,
                    "learning_rate": learning_rate,
                    "cost": cost,
                    "rSquare": rSquare,
                    "meanAbErr": meanAbErr,
                    "meanSqErr": meanSqErr,
                    "rootMeanSqErr": rootMeanSqErr,
                }
            )

    # Convert results to dataframe
    results = pd.DataFrame(results)

    # Sort results by MSE
    results = results.sort_values("meanSqErr", ascending=True)

    # Return best hyperparameters
    return (
        int(results.iloc[0]["iterations_nb"]),
        results.iloc[0]["learning_rate"],
        results,
    )



#Minimize redundant object creation
#Avoid unnecessary computation
#Reduce memory footprint by limiting stored data to only what's needed
#Early stopping for bad hyperparameters (optional but powerful)
#Parallezation : Optional: You can parallelize inner loop with joblib, concurrent.futures, or multiprocessing	
def computeOptimalParameters(x_train, x_test, y_train, y_test, model_type):
    # Hyperparameter space
    iterations_numbers = [200, 300, 400, 500, 1000, 2000, 3000, 5000, 7000, 10000, 20000, 50000]
    learning_rates = [0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005]

    np.random.seed(0)

    # Track only the best result
    best_result = None
    best_mse = float("inf")
    all_results = []

    # Pre-instantiate the model constructor map to avoid repeated if-else checks
    model_map = {
        "linear_2": LinearRegressionCustomModel2,
        "quadratic": QuadraticRegressionCustomModel,
        "degree_5": PolynomialDegree5RegressionCustomModel,
        "default": SubModel1
    }

    # Use the appropriate model class
    ModelClass = model_map.get(model_type, model_map["default"])
    x_train.head()
    for iterations_nb in iterations_numbers:
        for learning_rate in learning_rates:
            model = ModelClass(learning_rate, iterations_nb)

            # Fit model
            theta, cost = model.fit(x_train, y_train)

            # Skip invalid model
            if theta is None or np.isnan(cost) or np.isinf(cost):
                continue

            # Evaluate model
            r2, mae, mse, rmse = model.metrics(x_test, y_test)

            result = {
                "iterations_nb": iterations_nb,
                "learning_rate": learning_rate,
                "cost": cost,
                "rSquare": r2,
                "meanAbErr": mae,
                "meanSqErr": mse,
                "rootMeanSqErr": rmse,
            }
            all_results.append(result)

            # Update best only if necessary
            if mse < best_mse:
                best_mse = mse
                best_result = result

    # If no valid results
    if not all_results:
        return None, None, pd.DataFrame()

    # Create DataFrame only once
    results_df = pd.DataFrame(all_results).sort_values("meanSqErr", ascending=True)

    return (
        int(best_result["iterations_nb"]),
        best_result["learning_rate"],
        results_df,
    )
# Function to prepare data (split data into train and test sets, and standardise data)
def prepare_data(x, y, train_size):
    # Split data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)

    # standardise data
    x_train = (x_train - x_train.mean()) / x_train.std()  # standardise train set
    x_test = (x_test - x_test.mean()) / x_test.std()  # standardise test set

    #  Return train and test sets
    return x_train, x_test, y_train, y_test