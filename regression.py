# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
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


# AdaBoost regresyon modelini oluşturma
ada_regressor = AdaBoostRegressor(n_estimators=100, learning_rate=1.0, random_state=42)
ada_regressor.fit(X_train, y_train)

# Eğitim ve test verileri üzerinde tahmin yapma
y_train_pred = ada_regressor.predict(X_train)
y_test_pred = ada_regressor.predict(X_test)

# Eğitim ve test MSE değerini hesaplama
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Eğitim ve test R-kare değerini hesaplama
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Eğitim MSE:", train_mse)
print("Test MSE:", test_mse)
print("Eğitim R-kare:", train_r2)
print("Test R-kare:", test_r2)