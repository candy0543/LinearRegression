from Library.SubModel1 import SubModel1
import numpy as np


# Extend the LinearRegressionCustomModel1 class to implement a polynomial regression model of degree 5
class PolynomialDegree5RegressionCustomModel(SubModel1):
    # Add polynomial terms up to degree 5 to the dataset
    def __add_degree_5_terms(self, X):
        # Square each feature
        X_deg_2 = np.power(X, 2)
        X_deg_3 = np.power(X, 3)
        X_deg_4 = np.power(X, 4)
        X_deg_5 = np.power(X, 5)

        # Concatenate the original features and their polynomial values
        return np.hstack((X, X_deg_2, X_deg_3, X_deg_4, X_deg_5))

    def fit(self, x_train, y_train):
        # Transform x_train to include polynomial terms up to degree 5
        x_train_transformed = self.__add_degree_5_terms(x_train)

        # Call the fit method of the parent class (SubModel1)
        theta, cost = super().fit(x_train_transformed, y_train)

        return theta, cost

    def predict(self, x_test):
        # Transform x_test to include polynomial terms up to degree 5
        x_test_transformed = self.__add_degree_5_terms(x_test)

        # Call the predict method of the parent class
        return super().predict(x_test_transformed)