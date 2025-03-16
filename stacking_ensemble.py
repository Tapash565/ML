import numpy as np
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from classifiers import classifier

class StackingEnsemble(classifier):
    def __init__(self, base_models,meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.trained_base_models = []

    def fit(self, X, y):
        meta_features = []
        for model in self.base_models:
            model.fit(X, y)
            self.trained_base_models.append(model)
            meta_features.append(model.predict(X))

        meta_features = np.column_stack(meta_features)
        self.meta_model.fit(meta_features, y)
    
    def predict(self, X):
        meta_features = np.column_stack([model.predict(X) for model in self.trained_base_models])
        return self.meta_model.predict(meta_features)