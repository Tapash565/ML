from decision_tree import DecisionTree
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score

class RandomForest():
    def __init__(self,n_trees=10,max_depth=10,min_samples_split=2,n_features=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []
    
    def fit(self,X,y):
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,min_samples_split=self.min_samples_split,n_features=self.n_features)
            X_sample,y_sample = self._bootstrap_samples(X,y)
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self,X,y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples,n_samples,replace=True)
        return X[idxs],y[idxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self,X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions,0,1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_preds])
        return predictions
    
    def accuracy(self, y_true, y_pred):
        """
        Compute the accuracy of the model.
        """
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