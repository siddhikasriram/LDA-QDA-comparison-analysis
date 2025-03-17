# -*- coding: utf-8 -*-

"""
INFSY 566- Discriminant Analysis using Decision Trees
Entropy and Gini Decision Trees, Plotting Decision Tree
Printing Confusion Matrix
Printing Classification Report (See Page 166 of your course text)
Printing Confusion matrix using Seaborn
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (train_test_split)

# select a file from data directory
#data_dir='D:'
df=pd.read_csv('simClass.csv', header=None)
print (df.head()) # see first six rows to check everything

# Rename column titles
df.columns = ['v1', 'v2', 'v3', 'v4', 'v5', 'decision']
print (df.head()) # see first six rows to check everything

# Define independent variables and class variables

X = df[['v1', 'v2', 'v3', 'v4', 'v5']]
y = df['decision']


from sklearn.tree import DecisionTreeClassifier
# ID3 Decision tree
idt = DecisionTreeClassifier(criterion='entropy')
idt.fit(X,y)
from sklearn import tree
tree.plot_tree(idt)
plt.show()

from sklearn.model_selection import RepeatedStratifiedKFold

#Define method to evaluate model
cvv = RepeatedStratifiedKFold(n_splits=10, random_state=1)

from sklearn.model_selection import cross_val_score
#evaluate model
scores = cross_val_score(idt, X, y, scoring='accuracy', cv=cvv, n_jobs=-1)
print('Entropy DT Results:',np.mean(scores)) 


## CART Decision Tree

#Decision Tree using Gini Index (CART)
dt = DecisionTreeClassifier()
dt.fit(X,y)

#Define method to evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

#evaluate model
scores2 = cross_val_score(dt, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('CART DT Results:',np.mean(scores2)) 


# Confusion Matrix

from sklearn.metrics import confusion_matrix
#Create Training and Test Dataset
(X_train, X_test, y_train, y_test)=train_test_split(X,y,random_state=1)

dt.fit(X_train, y_train) #learn decision tree
y_pred=dt.predict(X_test) # predict test dataset

confusion=confusion_matrix(y_test,y_pred)#,labels=[0,1])
print(confusion) # Column is Actual {0,1} and Row Title is Predicted {0,1}

# Classification Report
from sklearn.metrics import classification_report
print (classification_report(y_test,y_pred))

#Pretty Confusion Matrix using Heatmap and Seaborn

#Heatmap of confusion matrix using seaborn
confusion_df=pd.DataFrame(confusion, index=range(2), columns=range(2))
import seaborn as sns
axes=sns.heatmap(confusion_df, annot=True)
axes.set_xlabel('Predicted')
axes.set_ylabel('Actual')
