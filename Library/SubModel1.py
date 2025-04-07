import numpy as np
from sklearn import metrics


# Class to perform linear regression with gradient descent algorithm - Model 1
# y = θ_0 + θ_1 * x1 + θ_2 * x2 + ... + θ_n * xn
class SubModel1:
    def __init__(self, learning_rate, iterations_nb):
        self.learning_rate = learning_rate  # Learning rate
        self.iterations_nb = iterations_nb  # Number of iterations
        self.theta = None

    # Method to fit the model to the training data
    def fit(self, x_train, y_train):
        # Add column of ones to x_train - Bias / Intercept
        x_train = np.column_stack((np.ones(x_train.shape[0]), x_train))

        # Number of training samples and number of features
        n, r = x_train.shape

        # Initialise theta with a vector of zeros
        self.theta = np.zeros(r)

        # Initialise variables for convergence check
        prev_cost = float("inf")  # Set initial previous cost to infinity
        tolerance = 1e-6  # Convergence tolerance
        gradient_threshold = 1e5  # Gradient clipping threshold

        # Gradient descent algorithm
        for _ in range(self.iterations_nb):
            # Compute predictions (y_pred = x_train * theta)
            y_pred = x_train.dot(self.theta)

            # Compute error (difference between predictions and actual values)
            error = y_pred - y_train

            # Compute gradient
            gradient = (2 / n) * error.dot(x_train)

            # Gradient clipping to prevent divergence (if gradient is too large)
            gradient = np.clip(gradient, -gradient_threshold, gradient_threshold)

            # Update parameters
            self.theta -= self.learning_rate * gradient

            # Compute cost (mean squared error)
            cost = (1 / n) * np.sum((y_pred - y_train) ** 2)

            # Convergence check
            if abs(prev_cost - cost) < tolerance:
                break

            # Update previous cost
            prev_cost = cost

        # Return None if training was not successful
        if np.isnan(cost):
            return None, float("inf")  # Return None for theta and infinity for cost

        return self.theta, cost

    # Method to predict the output for a given input
    def predict(self, x_test):
        # Add column of ones to x_test - Bias / Intercept
        x_test = np.column_stack((np.ones(x_test.shape[0]), x_test))

        # Return predictions with interaction term
        y_pred = x_test.dot(self.theta)  # Compute predictions
        assert (
            y_pred.shape[0] == x_test.shape[0]
        ), "Mismatch in prediction shape"  # Check if the shape of the prediction is correct

        return y_pred

    # Method to compute the metrics of the model
    def metrics(self, x_test, y_test):
        # Compute predictions
        y_pred = self.predict(x_test)

        # Compute metrics
        rSquare = metrics.r2_score(y_test, y_pred)  # R squared
        meanAbErr = metrics.mean_absolute_error(y_test, y_pred)  # Mean absolute error
        meanSqErr = metrics.mean_squared_error(y_test, y_pred)  # Mean squared error
        rootMeanSqErr = np.sqrt(
            metrics.mean_squared_error(y_test, y_pred)
        )  # Root mean squared error

        return rSquare, meanAbErr, meanSqErr, rootMeanSqErr