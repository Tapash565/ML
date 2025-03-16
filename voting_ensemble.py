import numpy as np
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from classifiers import classifier

class VotingEnsemble(classifier):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])
        final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
        return final_preds