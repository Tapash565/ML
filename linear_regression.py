import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error
from regression import Regression

class LinearRegression(Regression):
    def __init__(self,learning_rate = 0.01, n_iterations = 100,lamda = 0,alpha = 0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.lamda = lamda
        self.alpha = alpha

    def fit(self, x_train, y_train):
        x_train = np.column_stack((np.ones(len(x_train)), x_train))
        self.weights = np.zeros(x_train.shape[1])
        for i in range(self.n_iterations):
            m = len(x_train)
            y_pred = x_train.dot(self.weights)
            diff = y_pred - y_train
            grad = (1/ m) * np.dot(x_train.T,diff) + (self.lamda/m) * (self.weights * self.alpha + (1 - self.alpha) * np.sign(self.weights))
            self.weights -= self.learning_rate * grad

    def predict(self, x_test):
        x_test = np.column_stack((np.ones(len(x_test)), x_test))
        return np.dot(x_test, self.weights)