import numpy as np
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from classifiers import classifier

class KNN(classifier):
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