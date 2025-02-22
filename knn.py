import numpy as np
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class KNN():
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            # Calculate the distances to all training points
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Get the indices of the k-nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            # Extract the labels of the k-nearest neighbors
            k_neighbor_labels = np.array(self.y_train)[k_indices]
            # Append the most common label to predictions
            predictions.append(np.bincount(k_neighbor_labels).argmax())
        return np.array(predictions)
    
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
        sns.heatmap(confusion_matrix(y_true,y_pred, normalize='true'),annot=True)
        plt.show()