import numpy as np
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from classifiers import classifier

class LogisticRegression(classifier, BaseEstimator, ClassifierMixin):
    def __init__(self,lr = 0.01, n_iters = 100,lamda = 0,alpha = 0):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.lamda = lamda
        self.alpha = alpha

    def sigmoid(self,z):
        sig = 1/(1+np.exp(-z))
        return sig
    
    def binary_cross_entropy(self, y, predictions):
        m = len(y)
        loss = -1/m * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        return loss
    
    def fit(self, X,y):
        X = np.column_stack((np.ones(len(X)), X))
        self.weights = np.ones(X.shape[1])
        for _ in range(self.n_iters):
            m = len(X)
            z = X.dot(self.weights)
            y_pred = self.sigmoid(z)
            diff = y_pred - y
            grad = (1/ m) * np.dot(X.T,diff) + self.alpha * (self.l1_ratio * np.sign(self.weights) + (1 - self.l1_ratio) * self.weights)
            self.weights -= self.lr * grad
    
    def predict(self, x_test):
        x_test = np.column_stack((np.ones(len(x_test)), x_test))
        pred = self.sigmoid(np.dot(x_test, self.weights))
        return (pred >= 0.5).astype(int)