import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,confusion_matrix
from collections import Counter
from classifiers import classifier
from sklearn.preprocessing import LabelEncoder

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None, counts=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.counts = counts  # Store counts of each class at leaf nodes
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree(classifier):
    def __init__(self,min_samples_split=2, max_depth=100, n_features=None,criterion="entropy"):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.criterion = criterion
        self.classes_ = None # Store the classes encountered during fitting

    def fit(self,X,y):
        self.classes_, y = np.unique(y, return_inverse=True) # Store the classes
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self._grow_tree(X,y)

    def _grow_tree(self,X,y,depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        #check the stopping criteria
        if (depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value, counts = self._most_common_label(y, return_counts=True)
            return Node(value=leaf_value, counts=counts)

        feat_idxs = np.random.choice(n_feats,self.n_features,replace=False)
        #find the best split
        best_feature, best_thresh = self._best_split(X,y,feat_idxs)

        # create child nodes
        left_idxs,right_idxs = self._split(X[:,best_feature],best_thresh)
        left = self._grow_tree(X[left_idxs,:],y[left_idxs],depth+1)
        right = self._grow_tree(X[right_idxs,:],y[right_idxs],depth+1)

        return Node(best_feature,best_thresh,left,right)

    def _best_split(self,X,y,feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None,None

        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                # calculate the informatin gain
                gain = self._information_gain(y,X_column,thr)
                if gain > best_gain and gain > 1e-6:
                    best_gain, split_idx, split_threshold = gain, feat_idx, thr
        return split_idx, split_threshold
    
    def _information_gain(self, y,X_column,threshold):
        """Calculate the Information Gain using Gini index or Entropy"""
        if self.criterion == "entropy":
            parent_impurity = self._entropy(y)
        else:
            parent_impurity = self._gini(y)
        # create children
        left_idx,right_idx = self._split(X_column,threshold)

        if len(y[left_idx])==0 or len(y[right_idx])==0:
            return 0
        # calculate avg weighted entropy of children

        n = len(y)
        n_l,n_r = len(left_idx),len(right_idx)
        if self.criterion == "entropy":
            e_l,e_r = self._entropy(y[left_idx]),self._entropy(y[right_idx])
        else:
            e_l,e_r = self._gini(y[left_idx]),self._gini(y[right_idx])
        child_impurity = (n_l/n)*e_l + (n_r/n)*e_r
        # Calculate the IG
        information_gain = parent_impurity - child_impurity
        return information_gain
    
    def _split(self,X_column,split_thresh):
        left_idxs = np.argwhere(X_column<=split_thresh).flatten()
        right_idxs = np.argwhere(X_column>split_thresh).flatten()
        return left_idxs,right_idxs

    def _entropy(self,y):
        _, counts = np.unique(y,return_counts=True)
        ps = counts/len(y)
        return -np.sum([p*np.log2(p + 1e-15)for p in ps if p>0])
    
    def _gini(self,y):
        _, counts = np.unique(y,return_counts=True)
        ps = counts/len(y)
        return 1 - np.sum([p ** 2 for p in ps])

    def _most_common_label(self, y, return_counts=False):
        if len(y) == 0:  # Prevent index error
            return None
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        if return_counts:
            return value, counter
        else:
            return value

    def predict(self,X):
        y_pred_encoded = np.array([self._traverse_tree(x,self.root) for x in X])
        y_pred = self.classes_[y_pred_encoded]
        return y_pred
    
    def _traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)
    
    def predict_proba(self, X):
        """Returns class probabilities for each sample in X."""
        probs = []
        for x in X:
            probs.append(self._traverse_tree_proba(x, self.root))
        return np.array(probs)

    def _traverse_tree_proba(self, x, node):
        if node.is_leaf_node():
            probabilities = np.zeros(len(self.classes_))
            if node.counts:
                total = sum(node.counts.values())
                for i, c in enumerate(self.classes_):
                    probabilities[i] = node.counts.get(c, 0) / total if total > 0 else 0
            return probabilities

        if x[node.feature] <= node.threshold:
            return self._traverse_tree_proba(x, node.left)
        return self._traverse_tree_proba(x, node.right)