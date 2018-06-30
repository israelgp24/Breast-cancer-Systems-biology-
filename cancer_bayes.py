# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 12:45:25 2018

@author: toshiba
"""

"""
Created on Fri Jun 29 11:19:48 2018

@author: toshiba
"""

#Load the library with the breast cancer dataset

from sklearn.datasets import load_breast_cancer
# Load scikit's random forest classifier library
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import RandomForestClassifier
# Load pandas
import pandas as pd
# Load numpy
import numpy as np
# Asignamos un random seed
np.random.seed(0)
# Creas un objeto llamado cancer 
cancer=load_breast_cancer()
# Creas una dataframe con 30 feature variables
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)


# Agregamos una nueva columna con the species names,Esto es lo que intentaremos predecir
df['species'] = pd.Categorical.from_codes(cancer.target, cancer.target_names)


#print(df.head())

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

# View the top 5 rows
#df.head()
#print(df.head())

# Creamos 2 nuevas dataframe , una con the training rows, otra con the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]
# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

# Create a list of the feature column's names
features = df.columns[:30]      
# View features
#print(features)
y = pd.factorize(train['species'])[0]
y_test = pd.factorize(test['species'])[0]

#print
#Creamos un Gaussiano
clf=GaussianNB()

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train[features], y)

predic=clf.predict(test[features])
error=clf.score(test[features],y_test,sample_weight=None)
print(error)
#print(clf.predict(test[features]))

predict_porcentaje=clf.predict_proba(test[features])[0:50] 

#print(clf.predict_proba(test[features])[0:50] )

preds = cancer.target_names[clf.predict(test[features])]

print(preds[0:5])

print(test['species'].head())

 # Create confusion matrix
matrix=pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])

