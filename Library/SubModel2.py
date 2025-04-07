import numpy as np
from sklearn import metrics


# Class to perform linear regression with gradient descent algorithm - Model 2
# y = θ_0 + θ_1 * x1 + θ_2 * x2 + θ_3 * (x1 * x2)
class LinearRegressionCustomModel2:
    def __init__(self, learning_rate, iterations_nb):
        self.learning_rate = learning_rate  # Learning rate
        self.iterations_nb = iterations_nb  # Number of iterations
        self.theta = None

    # Method to fit the model to the training data
    def fit(self, x_train, y_train):
        # Convert DataFrame to numpy array
        x_train = x_train.values

        # Add column of ones to x_train - Bias / Intercept
        interaction_term = (
            x_train[:, 0] * x_train[:, 1]
        )  # Interaction term between x1 and x2
        x_train = np.column_stack(
            (np.ones(x_train.shape[0]), x_train, interaction_term)
        )

        # Number of training samples and number of features
        n, r = x_train.shape

        # Initialise theta with a vector of zeros
        self.theta = np.zeros(r)

        # Gradient descent algorithm
        for _ in range(self.iterations_nb):
            # Compute predictions
            y_pred = x_train.dot(self.theta)

            # Compute error (difference between predictions and actual values)
            error = y_pred - y_train

            # Compute gradient
            gradient = (2 / n) * error.dot(x_train)

            # Update parameters
            self.theta -= self.learning_rate * gradient

            # Compute cost (mean squared error)
            cost = (1 / n) * np.sum((y_pred - y_train) ** 2)

        return self.theta, cost

    # Method to predict the output for a given input
    def predict(self, x_test):
        # Add column of ones to x_test - Bias / Intercept
        x_test = np.column_stack((np.ones(x_test.shape[0]), x_test))

        # Return predictions with interaction term
        y_pred = (
            self.theta[0]
            + self.theta[1] * x_test[:, 1]
            + self.theta[2] * x_test[:, 2]
            + self.theta[3] * (x_test[:, 1] * x_test[:, 2])
        )
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