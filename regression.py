# -*- coding: utf-8 -*-
"""
Created on Mon May  1 15:54:16 2023

@author: kubra
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

regression_df=pd.read_csv('data_preprocessing/reg_dataset.csv')


X=regression_df.drop(['Like Count'],axis=1)
y=regression_df['Like Count']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=100)

# Random Forest Regresyon modelini oluşturun
regressor = RandomForestRegressor(n_estimators=100, random_state=0)

# Modeli eğitin
regressor.fit(X_train, y_train)

# Test verileri üzerinde tahmin yapın
y_pred = regressor.predict(X_test)

# Model performansını değerlendirin
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared Score:', r2)