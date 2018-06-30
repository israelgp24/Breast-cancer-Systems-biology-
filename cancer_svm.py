# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:29:56 2018

@author: toshiba
"""
#Load the library with the breast cancer dataset

from sklearn.datasets import load_breast_cancer
# Load scikit's random forest classifier library
import matplotlib.pyplot as plt

from tools import plot_decision_regions

from sklearn import svm
#from sklearn.ensemble import RandomForestClassifier
# Load pandas
import pandas as pd
# Load numpy
import numpy as np
# Set random seed

from tools import plot_decision_regions

np.random.seed(0)
# Create an object called cancer with the iris data
cancer=load_breast_cancer()
# Create a dataframe with the 30 feature variables
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

#df.to_csv('cancer.csv')

# Add a new column with the species names, this is what we are going to try to predict
df['species'] = pd.Categorical.from_codes(cancer.target, cancer.target_names)

#print(df.head())

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]
# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

# Create a list of the feature column's names
features = df.columns[:30]      

y = pd.factorize(train['species'])[0]
y_test = pd.factorize(test['species'])[0]


# Create a SVM
clf=svm.SVC(C=1,kernel='linear')


# Train the Classifier to take the training features and learn how they relate

# PREDECIR SOLO 2 ATRIBUTOS
#features=["worst perimeter","mean concave points"]
clf.fit(train[features], y)

predic=clf.predict(test[features])
error=clf.score(test[features],y_test,sample_weight=None)
print(error)


preds = cancer.target_names[clf.predict(test[features])]

print(preds[0:5])

print(test['species'].head())

 # Create confusion matrix
matrix=pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])

