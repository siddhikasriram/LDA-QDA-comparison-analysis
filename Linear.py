# -*- coding: utf-8 -*-
"""
INFSY 566- Discriminant Analysis with dataset containing missing values
Linear Discriminant Analysis, Training and Test Sampling
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (train_test_split)


df=pd.read_csv('MisssimClass.csv', header=None) # use it when file does not have headers
# variables are now df[0]...df[5]
#print(df[0]) #print first column
# Rename column titles
df.columns = ['v1', 'v2', 'v3', 'v4', 'v5', 'decision']
print (df.head()) # see first six rows to check everything


# How many missing values are there?

df.isnull().sum()
 
# Check their distributions to decide if mean median or mode should be used to fill missing values
import seaborn as sns
#
# Distribution plot
#
sns.displot(df.v2)

print(np.mean(df.v2))#  Check mean value and compare it to histogram

# Fill missing values using mean imputation and store it in new data file (ndf)

ndf=df.fillna(df.mean())

# Check if there are any missing values left
ndf.isnull().sum()

# Define independent variables and class variables for new data file
X = ndf[['v1', 'v2', 'v3', 'v4', 'v5']]
y = ndf['decision']

# split dataset into training and testing 70-30 ratio

X_train, X_test, y_train, y_test=train_test_split (X,y, test_size=0.3)# add fourth parameter random_state=10 for seeded random number generation
print('size of test dataset:',len(X_test), ' size of training dataset: ', len(X_train))

#Fit the LDA model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train) #learn Discriminant Function

# Print Training and Test Accuracies
result1 = model.score(X_train, y_train)
print(("LDA Training Accuracy: %.3f%%") % (result1*100))

result = model.score(X_test, y_test)
print(("LDA Test Accuracy: %.3f%%") % (result*100.0))








