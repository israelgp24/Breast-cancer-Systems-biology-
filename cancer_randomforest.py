# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 10:46:47 2018

@author: toshiba
"""

"""
Created on Fri Jun 29 11:19:48 2018

@author: toshiba
"""

#Load the library with the breast cancer dataset

from sklearn.datasets import load_breast_cancer
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
# Load pandas
import pandas as pd
# Load numpy
import numpy as np


import matplotlib.pyplot as plt

# Asiganar un random seed
np.random.seed(0)
# Creas un objeto llamado cancer 
cancer=load_breast_cancer()

# Creas una dataframe con 30 feature variables
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)


#guardamos la data
#df.to_csv('cancer.csv')


# Agregamos una nueva columna con the species names,Esto es lo que intentaremos predecir
df['species'] = pd.Categorical.from_codes(cancer.target, cancer.target_names)


# separamos la data.
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75



# Creamos 2 nuevas dataframe , una con the training rows, otra con the test rows
train, test = df[df['is_train']==True], df[df['is_train']==False]


# Mostramos los nuemros de observaciones para the test y training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))

# Creamos una lista de the feature column's names
features = df.columns[:30]      


y = pd.factorize(train['species'])[0]
y_test = pd.factorize(test['species'])[0]
#print


# Creamos un random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0,n_estimators=600)

# Train the Classifier to take the training features and learn how they relate
# to the training y (the species)

#features=["worst perimeter","mean concave points"]
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

 # Creamos una matrix de confusión
matrix=pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])

#print(matrix)

# View a list of the features and their importance scores
importance= list(zip(train[features], clf.feature_importances_))

#print(importance)


n_features = cancer.data.shape[1]
plt.barh(range(n_features), clf.feature_importances_, align='center')
plt.yticks(np.arange(n_features), cancer.feature_names)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()