import numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score, confusion_matrix
from classifiers import classifier

class GaussianNB(classifier):
    
    def fit(self, X, y):
        """
        Fit the model to the training data by calculating the mean and variance
        of each feature for each class.
        """
        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.parameters = {}
        eps = 1e-4
        
        for c in self.classes:
            # Extract all rows for a specific class
            X_c = X[y == c]
            # Store mean and variance for each feature
            self.parameters[c] = {
                "mean": X_c.mean(axis=0),
                "var": X_c.var(axis=0, ddof=1)  # Use unbiased variance
            }
        self.class_counts = np.array([np.sum(y == c) for c in self.classes])
        self.class_priors = self.class_counts / len(y)
        self.log_priors = np.log(np.clip(self.class_priors, eps, 1.0))

    def _calculate_log_likelihood(self, mean, var, x):
        """
        Compute the log of the Gaussian likelihood for a given feature value.
        Handles numerical stability by working in the log space.
        """
        eps = 1e-4  # Small constant to prevent division by zero
        var = np.maximum(var, eps)  # Ensure variance is non-zero
        coeff = -0.5 * np.log(2.0 * np.pi * var)
        exponent = -0.5 * ((x - mean) ** 2) / var
        return coeff + exponent  # Return log-likelihood
 
    def _classify(self, X):
        """
        Classify multiple samples by computing the posterior probabilities for each class.
        """
        log_likelihoods = np.zeros((X.shape[0], len(self.classes)))

        for i, params in self.parameters.items():
            log_likelihoods[:, i] = np.sum(self._calculate_log_likelihood(params["mean"], params["var"], X), axis=1)

        log_posteriors = self.log_priors + log_likelihoods
        return self.classes[np.argmax(log_posteriors, axis=1)]


    def predict(self, X):
        """
        Predict the class labels for the given samples.
        """
        return self._classify(X)

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    data = datasets.load_breast_cancer()
    X,y = data.data,data.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    clf = GaussianNB()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    clf.accuracy(y_test,y_pred)