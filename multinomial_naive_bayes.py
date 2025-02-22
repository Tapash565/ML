import numpy as np
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin

class MultinomialNB(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        """
        Fits the Naive Bayes model to the training data.

        Args:
            X: A 2D NumPy array of features.
            y: A 1D NumPy array of labels.
        """
        self.classes, self.class_counts = np.unique(y, return_counts=True)
        self.class_priors = dict(zip(self.classes, self.class_counts / len(y)))


        self.feature_counts = {}
        for c in self.classes:
            X_c = X[y == c]
            self.feature_counts[c] = {}
            for feature_idx in range(X.shape[1]):
                feature_values, feature_counts = np.unique(X_c[:, feature_idx], return_counts=True)
                self.feature_counts[c][feature_idx] = dict(zip(feature_values, feature_counts))

    def predict(self, X):
        """
        Predicts the class for each sample in X.

        Args:
            X: A 2D NumPy array of features.

        Returns:
            A 1D NumPy array of predicted classes.
        """
        return np.argmax(self.predict_proba(X), axis=1)
    
    def predict_proba(self, X):
        probs = []
        for x in X:
            class_probs = []
            for c in self.classes:
                prob = self.class_priors.get(c, 0)
                for feature_idx, feature_value in enumerate(x):
                    if feature_value in self.feature_counts[c][feature_idx]:
                        feature_count = self.feature_counts[c][feature_idx].get(feature_value, 0) + 1
                        class_count = self.class_counts[np.where(self.classes == c)[0][0]] + len(self.feature_counts[c][feature_idx])
                        prob *= feature_count / class_count
                    else:
                        prob *= 0  # Handle unseen features
                class_probs.append(prob)
            probs.append(np.array(class_probs) / sum(class_probs))
        return np.array(probs)
    
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