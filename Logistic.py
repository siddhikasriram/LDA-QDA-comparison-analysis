# -*- coding: utf-8 -*-
"""
Quadratic Discriminant Analysis, Stratified K-Fold Sampling, ROC-Area Under the Curve,
Logistic Regression.

ROC-AUC will only work for technique that outputs probability so do not use it for Discriminant
Analysis
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold


df=pd.read_csv('simClass.csv', header=None) # use it when file does not have headers
# variables are now df[0]...df[5]
#print(df[0]) #print first column
# Rename column titles
df.columns = ['v1', 'v2', 'v3', 'v4', 'v5', 'decision']
print (df.head()) # see first six rows to check everything

# Define independent variables and class variables

X = df[['v1', 'v2', 'v3', 'v4', 'v5']]
y = df['decision']


 
# Cross Validation using 10 fold sampling. n_repeats=3 means you repeat 10 folds 3 times

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#Fit the LDA model
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis  
#Fit the QDA model
model2 = QuadraticDiscriminantAnalysis()
model2.fit(X, y)

#evaluate model
from sklearn.model_selection import cross_val_score
scores2 = cross_val_score(model2, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print(np.mean(scores2)) 

#define new observation
new = [14, 9, 5.1, 7, 4]

#predict which class the new observation belongs to
print ('QDA Prediction: Class ',model2.predict([new]))


# Logistic Regression with Area Under the Curve-may not work with Iris due to dataset loading
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
from sklearn.metrics import roc_auc_score
print(" ROC-AUC score: %.3f" % roc_auc_score(y, clf.predict_proba(X)[:, 1]))

# Neural Network
from sklearn.neural_network import MLPClassifier
# 3 hidden layers with 8 nodes in each layer. Sigmoid (logistic) function
nn=MLPClassifier(activation='logistic', hidden_layer_sizes=(3,8),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01, max_iter=10000)
nn.fit(X,y)
scores3 = cross_val_score(nn, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('NN Cross Validation Score: ',np.mean(scores3)) 

