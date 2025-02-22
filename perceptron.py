import numpy as np
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def unit_step_func(x):
    return np.where(x>0,1,0)

class Perceptron():
    def __init__(self,lr=0.01,n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None
    
    def fit(self,X,y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.where(y > 0,1,0)

        for _ in range(self.n_iters):
            for idx,x in enumerate(X):
                z = np.dot(x,self.weights) + self.bias
                y_pred = self.activation_func(z)

                update = self.lr * (y_[idx]-y_pred)
                self.weights += update * x
                self.bias += update

    def predict(self,X):
        z = np.dot(X,self.weights) + self.bias
        y_pred = self.activation_func(z)
        return y_pred
    
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