import numpy as np,matplotlib.pyplot as plt
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
class SVM:
    def __init__(self,learning_rate=0.001,lambda_param=0.01,n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        y = np.where(y == 0,-1,1)
        n_samples, n_features = X.shape
        y_ = np.where(y<=0,-1,1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i,self.weights) - self.bias) >=1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights - y[idx] * x_i)
                    self.bias -= self.lr * y_[idx]

    def predict(self,X):
        approx = np.dot(X,self.weights) - self.bias
        return np.sign(approx)

    def accuracy(self, y_true, y_pred):
        """
        Compute the accuracy of the model.
        """
        y_true = np.where(y_true == 0,-1,1)
        unique_classes = np.unique(y_true)  # Get unique class labels
        avg_mode = 'weighted' if len(unique_classes) > 2 else None
        f1 = f1_score(y_true, y_pred,average=avg_mode)
        recall = recall_score(y_true, y_pred,average=avg_mode)
        precision = precision_score(y_true, y_pred,average=avg_mode)
        acc = accuracy_score(y_true,y_pred) *100
        print("Custom Accuracy")
        print(f"Accuracy: {acc:.2f}%")
        print(f"F1 score: {f1}")
        print(f"Recall: {recall}")
        print(f"Precision: {precision}")

    