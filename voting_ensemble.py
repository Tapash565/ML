import numpy as np
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score

class VotingEnsemble():
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])
        final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
        return final_preds
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