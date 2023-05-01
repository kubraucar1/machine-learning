#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 15:28:23 2023

@author: esmanur
"""

# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_extraction import text
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('dataset.csv')

# Drop unnecessary columns
data = data.drop(columns=['Id', "User Name","Date of Tweet"])

# Convert text features to lowercase
data['Text'] = data['Text'].str.lower()

# Separate features and target variable
X = data.drop(['Like Count'], axis=1)
y = data['Like Count']

my_stop_words = list(text.ENGLISH_STOP_WORDS.union(["book", "film"]))

#numeric featrue names
coulmn_names = list(X.columns)
coulmn_names.remove("Text")


# Preprocess text data
Tfidf_transformer = TfidfVectorizer(stop_words=my_stop_words)
count_transformer = CountVectorizer(stop_words=my_stop_words)
scaler = StandardScaler()

# Combine text and numeric features
preprocessor = ColumnTransformer(transformers=[
    ("T_Text", Tfidf_transformer, "Text"),
   
    ('C_Text', count_transformer, 'Text'),
    ('numeric', 'passthrough', coulmn_names)
])
# Fit and transform the preprocessor to the data
X_preprocessed = preprocessor.fit_transform(X)
X_preprocessed_df = pd.DataFrame(X_preprocessed.toarray(), columns=preprocessor.get_feature_names_out())
X_preprocessed_df.to_csv('tweets_in.csv')  
y.to_csv('tweets_out.csv')  

# Show the dataframe
print(X_preprocessed_df.head())
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed_df, y, test_size=0.4, random_state=42)
# Create a linear regression model
lr = LinearRegression()
# Fit the model to the training data
lr.fit(X_train, y_train)
# Predict on the test data
y_pred = lr.predict(X_test)
# Evaluate the model using R-squared score
r2 = r2_score(y_test, y_pred)
print(f"R-squared score: {r2:.3f}")