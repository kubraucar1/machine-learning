# -*- coding: utf-8 -*-
"""
Created on Mon May  1 15:54:16 2023

@author: kubra
"""

import pandas as pd
import numpy as np

#project

dataset= pd.read_excel("data_preprocessing/all_tweets.xlsx") 
#print("number of rows in the original dataset",len(dataset))
dataset.drop_duplicates(inplace=True)
#print("Number of rows in the cleaned dataset:",len(dataset))
pd.read_excel("data_preprocessing/all_tweets.xlsx") 
dataset.dropna()

like_column = 'Like Count'
comment_column = 'Comment Count'
retweet_column = 'retweet Count'
view_column = 'View count'

dataset[like_column]

def convert_k_m(column_name):
    for i in range(len(dataset)):
        deger = str(dataset.at[i, column_name])
        if 'K' in deger:
            deger = deger.replace('K', '')
            deger = float(deger) * 1000
            dataset.at[i, column_name] = int(deger)
        elif  'M' in deger:
            deger = deger.replace('M', '')
            deger = float(deger) * 1000000
            dataset.at[i, column_name] = int(deger)
        
convert_k_m(like_column)
convert_k_m(comment_column)
convert_k_m(retweet_column)
convert_k_m(view_column)


dataset[comment_column].fillna(0, inplace=True)
dataset[comment_column] = dataset[comment_column].astype(int)
dataset[like_column].fillna(0, inplace=True)
dataset[like_column] = dataset[like_column].astype(int)
dataset[view_column].fillna(0, inplace=True)
dataset[view_column] = dataset[view_column].astype(int)
dataset[retweet_column].fillna(0, inplace=True)
dataset[retweet_column] = dataset[retweet_column].astype(int)


text_column = dataset["Text"]
#kelime sayısını ayırma
def countOfWord(text):
    liste = []
    for x in range(0,len(text)):
        liste.append(len(text[x].split()))
    return liste  

countOfText = countOfWord(text_column)

dataset["countOfWords"] = countOfText

data_positive = pd.read_csv("data_preprocessing/PositiveWordsEng.csv")
data_negative = pd.read_csv("data_preprocessing/NegativeWordsEng.csv")
data_positive = data_positive["PositiveWords"]
data_negative = data_negative["NegativeWords"]

def search_tw(data_words,data_tw):
    liste = []
    for tweet in data_tw:
        count = 0 
        tweet = str(tweet).lower().split()
        for word in data_words:
            if word in tweet:
                count+=1
        liste.append(count)
    return liste 

countOfPositive = search_tw(data_positive,text_column)
countOfNegative = search_tw(data_negative,text_column)


dataset["countOfPositive"] = countOfPositive

dataset["countOfNegative"] = countOfNegative

#########################

print(dataset)
#classification_df=dataset.to_csv('cl_dataset.csv',index=False)
classification_df=pd.read_csv('data_preprocessing/cl_dataset.csv')

##kubra
"""
one-hot encoding işleleri classification için gerekli değil

one_hot_df = pd.get_dummies(dataset['Title'])
dataset['ID'] = range(1, len(dataset) + 1)
one_hot_df['ID'] = range(1, len(one_hot_df) + 1)


one_hot_df[one_hot_df == True] = "1"
one_hot_df[one_hot_df == False] = "0"

merged_df = pd.merge(dataset, one_hot_df, on='ID')

merged_df = merged_df.drop('ID',axis=1)
merged_df = merged_df.drop('Title',axis=1)
merged_df.to_csv("dataset.csv.",index=False)
dataset = pd.read_csv('dataset.csv')


dataset
"""
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