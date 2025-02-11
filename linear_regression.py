import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error
class LinRegression:
    def __init__(self,lr = 0.01, num_iters = 100,lamda = 1,alpha = 0.5):
        self.lr = lr
        self.num_iters = num_iters
        self.weights = None
        self.lamda = lamda
        self.alpha = alpha

    def fit(self, x_train, y_train):
        x_train = np.column_stack((np.ones(len(x_train)), x_train))
        self.weights = np.zeros(x_train.shape[1])
        for i in range(self.num_iters):
            m = len(x_train)
            y_pred = x_train.dot(self.weights)
            diff = y_pred - y_train
            grad = (1/ m) * np.dot(x_train.T,diff) + (self.lamda/m) * (self.weights * self.alpha + (1 - self.alpha) * np.sign(self.weights))
            self.weights -= self.lr * grad

    def predict(self, x_test):
        x_test = np.column_stack((np.ones(len(x_test)), x_test))
        return np.dot(x_test, self.weights)
    
    def accuracy(self,actual,predicted):
        r2 = r2_score(actual,predicted)
        mae = mean_absolute_error(actual,predicted)
        rmse = root_mean_squared_error(actual,predicted)
        print(f"R2 score: {r2}")
        print(f"Mean Absolute Error: {mae}")
        print(f"Root Mean Square Error: {rmse}")