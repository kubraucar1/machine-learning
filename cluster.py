#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 11:54:30 2023

@author: esmanur
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("data_preprocessing/reg_dataset.csv")

X = data.drop(["Like Count"], axis=1)
y = data["Like Count"]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(X)

outlier_detection = DBSCAN(eps=3.23, min_samples=5,metric="euclidean") #değiştirince azaldı orangedan baktım
clusters = outlier_detection.fit_predict(df_scaled)

#print(clusters)

outliers = np.where(clusters == -1)
print(outliers)


"""
plt.scatter(df_scaled[:, 2], df_scaled[:, 1])
plt.xlabel("Comment Count")
plt.ylabel("Lİke Count")
plt.show()

plt.scatter(df_scaled[:, 2], df_scaled[:, 1], c=clusters)
plt.xlabel("Comment Count")
plt.ylabel("Lİke Count")
plt.show()
"""

data_o = data.drop(outliers[0],axis = 0)

# K-means kümeleme
kmeans = KMeans(n_clusters=5, random_state=34,n_init=5).fit(df_scaled)

# Aykırı değerleri tespit etme
distances = kmeans.transform(df_scaled)
std_dev = np.std(distances, axis=0)
mean_dist = np.mean(distances, axis=0)

threshold = mean_dist + 2 * std_dev
outliers = np.where(np.any(distances > threshold, axis=1))[0]

print("Aykırı değerlerin sayısı:", len(outliers))
print(outliers)






