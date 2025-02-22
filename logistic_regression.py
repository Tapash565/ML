import numpy as np
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin

class LogRegression(BaseEstimator, ClassifierMixin):
    def __init__(self,lr = 0.01, num_iters = 100,lamda = 0,alpha = 0):
        self.lr = lr
        self.num_iters = num_iters
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
    
    def fit(self, x_train,y_train):
        x_train = np.column_stack((np.ones(len(x_train)), x_train))
        self.weights = np.ones(x_train.shape[1])
        for i in range(self.num_iters):
            m = len(x_train)
            z = x_train.dot(self.weights)
            y_pred = self.sigmoid(z)
            diff = y_pred - y_train
            grad = (1/ m) * np.dot(x_train.T,diff) + (self.lamda/m) * (self.alpha * self.weights + (1 - self.alpha) * np.sign(self.weights))
            self.weights -= self.lr * grad
    
    def predict(self, x_test):
        x_test = np.column_stack((np.ones(len(x_test)), x_test))
        pred = self.sigmoid(np.dot(x_test, self.weights))
        return (pred >= 0.5).astype(int)
    
    def accuracy(self, y_true, y_pred):
        """
        Compute the accuracy of the model.
        """
        unique_classes = np.unique(y_true)  # Get unique class labels
        avg_mode = 'weighted' if len(unique_classes) > 2 else None
        print("Custom Accuracy")
        print(f"Accuracy: {accuracy_score(y_true,y_pred) *100:.2f}%")
        print(f"F1 score: {f1_score(y_true, y_pred,average=avg_mode)}")
        print(f"Recall: {recall_score(y_true, y_pred,average=avg_mode)}")
        print(f"Precision: {precision_score(y_true, y_pred,average=avg_mode)}")
        sns.heatmap(confusion_matrix(y_true,y_pred, normalize='true'),annot=True,fmt='.2f')
        plt.show()