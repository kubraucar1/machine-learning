# -*- coding: utf-8 -*-
"""
Created on Mon May  1 15:54:16 2023

@author: kubra
"""

import pandas as pd
import numpy as np


classification_df=pd.read_csv('data_preprocessing/cl_dataset.csv')


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print(classification_df)

x=classification_df.drop(['Title'],axis=1)
y=classification_df['Title']
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

# Instantiate the random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
#Accuracy: 0.9578947368421052            last result 