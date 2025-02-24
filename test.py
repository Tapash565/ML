import numpy as np,matplotlib.pyplot as plt,pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from random_forest import RandomForest
from decision_tree import DecisionTree
from logistic_regression import LogRegression
from gaussian_naive_bayes import GaussianNB
from perceptron import Perceptron
from voting_ensemble import VotingEnsemble
from stacking_ensemble import StackingEnsemble
from knn import KNN
from multinomial_naive_bayes import MultinomialNB
from svm import SVM
from bagging_ensemble import BaggingEnsemble
import warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')



data = datasets.load_breast_cancer()
X,y = data.data,data.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=12)

# models = [RandomForest(),KNN(k=5),GaussianNB(),Perceptron(),LogRegression(),DecisionTree()]
# base_models = [GaussianNB(),DecisionTree()]
# meta_model = RandomForest()
clf = DecisionTree()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
clf.accuracy(y_test,y_pred)