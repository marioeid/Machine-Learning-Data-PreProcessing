
# Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing our data set
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:, :-1].values 
y=dataset.iloc[:,3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
simple_imputer=SimpleImputer(missing_values=np.nan,strategy='mean') # replace the missing values(NaN) by the mean for every column
simple_imputer=simple_imputer.fit(X[:,1:3]) # the upper bound is excluded fit for the indexes 1 and 2
X[:,1:3]=simple_imputer.transform(X[:,1:3]) # to transform the data

# Encoding Categorial Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# sklearn new format for one hot encoding check the medium website in the book marks
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X), dtype =np.float)

from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y) # the machine learning model will encoded automatically cause it's the dependant variable

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scalling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# you can scale your dummy variables or not based on the context and what you want to do
# with your data if you scalled it you will lose the knoweldge of the encding 
# but you may get better accuracy if you don't they will be already scaled for this model
# we will scale them
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# we don't to scale the dependant variable here cause it's classification problem 
# but in regression we will have to scale it as it will take wide range of numbers 

