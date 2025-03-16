import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix,f1_score,recall_score,accuracy_score,precision_score
import copy
from classifiers import classifier

class BaggingEnsemble(classifier):
    def __init__(self, base_model, n_estimators=10):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.classifiers = []
    def fit(self,X,y):
        for _ in range(self.n_estimators):
            model = copy.deepcopy(self.base_model)
            X_sample, y_sample = self._bootstrap_samples(X,y)
            model.fit(X_sample,y_sample)
            self.classifiers.append(model)

    def _bootstrap_samples(self,X,y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples,n_samples,replace=True)
        return X[idxs],y[idxs]
    
    def predict(self,X):
        predictions = np.array([model.predict(X) for model in self.classifiers])
        model_preds = np.swapaxes(predictions,0,1)
        predictions = np.array([self._most_common_label(pred) for pred in model_preds])
        return predictions
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common