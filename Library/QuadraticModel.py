from Library.SubModel1 import SubModel1
import numpy as np


# Extend the LinearRegressionCustomModel1 class to implement a quadratic regression model
class QuadraticRegressionCustomModel(SubModel1):
    # Function to add quadratic terms to the dataset
    def __add_quadratic_terms(self, X):
        # Square each feature
        X_squared = np.square(X)

        # Concatenate the original features and their squared values
        return np.hstack((X, X_squared))

    def fit(self, x_train, y_train):
        # Transform x_train to include quadratic terms
        x_train_transformed = self.__add_quadratic_terms(x_train)

        # Call the fit method of the parent class (SubModel1)
        theta, cost = super().fit(x_train_transformed, y_train)

        return theta, cost

    def predict(self, x_test):
        # Transform x_test to include quadratic terms
        x_test_transformed = self.__add_quadratic_terms(x_test)

        # Call the predict method of the parent class
        return super().predict(x_test_transformed)