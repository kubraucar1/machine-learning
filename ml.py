import pandas as pd
import numpy as np

#project

dataset= pd.read_excel("all_tweets.xlsx") 
print("number of rows in the original dataset",len(dataset))
dataset.drop_duplicates(inplace=True)
print("Number of rows in the cleaned dataset:",len(dataset))
pd.read_excel("all_tweets.xlsx") 
dataset.dropna()

like_column = 'Like Count'
comment_column = 'Comment Count'
retweet_column = 'retweet Count'
view_column = 'View count'

dataset[like_column]

for i in range(len(dataset)):
    deger = str(dataset.at[i, like_column])
    if 'K' in deger:
        deger = deger.replace('K', '')
        deger = float(deger) * 1000
        dataset.at[i, like_column] = int(deger)
    elif  'M' in deger:
        deger = deger.replace('M', '')
        deger = float(deger) * 1000000
        dataset.at[i, like_column] = int(deger)
        
for i in range(len(dataset)):
    deger = str(dataset.at[i, view_column])
    if 'K' in deger:
        deger = deger.replace('K', '')
        deger = float(deger) * 1000
        dataset.at[i, view_column] = int(deger)
    elif 'M' in deger:
        deger = deger.replace('M', '')
        deger = float(deger) * 1000000
        dataset.at[i, view_column] = int(deger)
        
dataset[view_column]
for i in range(len(dataset)):
    deger = str(dataset.at[i, comment_column])
    if 'K' in deger:
        deger = deger.replace('K', '')
        deger = float(deger) * 1000
        dataset.at[i,comment_column] = int(deger)
    elif 'M' in deger:
        deger = deger.replace('M', '')
        deger = float(deger) * 1000000
        dataset.at[i, comment_column] = int(deger)
for i in range(len(dataset)):
    deger = str(dataset.at[i, retweet_column])
    if 'K' in deger:
        deger = deger.replace('K', '')
        deger = float(deger) * 1000
        dataset.at[i,retweet_column] = int(deger)
    elif 'M' in deger:
        deger = deger.replace('M', '')
        deger = float(deger) * 1000000
        dataset.at[i, retweet_column] = int(deger)       

dataset[comment_column].fillna(0, inplace=True)
dataset[comment_column] = dataset[comment_column].astype(int)
dataset[like_column].fillna(0, inplace=True)
dataset[like_column] = dataset[like_column].astype(int)
dataset[view_column].fillna(0, inplace=True)
dataset[view_column] = dataset[view_column].astype(int)
dataset[retweet_column].fillna(0, inplace=True)
dataset[retweet_column] = dataset[retweet_column].astype(int)

dataset.to_excel("newdataset.xlsx",index=False)
newdataset= pd.read_excel("newdataset.xlsx") 
newdataset