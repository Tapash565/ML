import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error
from regression import Regression
from sklearn.linear_model import ElasticNet

class LinearRegression(Regression):
    def __init__(self,learning_rate = 0.01, n_iterations = 100,l1_ratio = 0,alpha = 0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.l1_ratio = l1_ratio
        self.alpha = alpha

    def fit(self, X, y):
        X = np.column_stack((np.ones(len(X)), X))
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.n_iterations):
            m = len(X)
            y_pred = X.dot(self.weights)
            diff = y_pred - y
            grad = (1/ m) * np.dot(X.T,diff) + self.alpha * (self.l1_ratio * np.sign(self.weights) + (1 - self.l1_ratio) * self.weights)
            self.weights -= self.learning_rate * grad

    def predict(self, x_test):
        x_test = np.column_stack((np.ones(len(x_test)), x_test))
        return np.dot(x_test, self.weights)