#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 11:54:30 2023

@author: esmanur
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

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

dbscn = DBSCAN(eps=3.227, min_samples=5,metric="euclidean") #değiştirince azaldı orangedan baktım
clusters = dbscn.fit_predict(df_scaled)

#print(clusters)

outliers = np.where(clusters == -1)
#print(outliers)


"""
plt.scatter(df_scaled[:, 2], df_scaled[:, 1])
plt.xlabel("Comment Count")
plt.ylabel("Lİke Count")
plt.show()

plt.scatter(df_scaled[:, 2], df_scaled[:, 1], c=clusters) #mor olanlar outlier
plt.xlabel("Comment Count")
plt.ylabel("Lİke Count")
plt.show()
"""

#data_o = data.drop(outliers[0],axis = 0)

# K-means kümeleme
kmeans = KMeans(n_clusters=5, random_state=1, n_init="auto").fit(df_scaled) #2 cluster'a sahipken daha yüksek değer veriyor
clusters_kmean = kmeans.fit_predict(df_scaled)


# Aykırı değerleri tespit etme
distances = kmeans.transform(df_scaled)
std_dev = np.std(distances, axis=0)
mean_dist = np.mean(distances, axis=0)

threshold = mean_dist + 2 * std_dev #Normal Distribution
outliers = np.where(np.any(distances > threshold, axis=1))[0]
"""
plt.scatter(df_scaled[:, 2], df_scaled[:, 1], c=clusters_kmean) #mor olanlar outlier
plt.xlabel("Comment Count")
plt.ylabel("Lİke Count")
plt.show()
"""
#print("Number of outliers:", len(outliers))
#print(outliers)

from sklearn import metrics
from sklearn.mixture import GaussianMixture

parameters=['full','tied','diag','spherical']
n_clusters=np.arange(1,21)
results_=pd.DataFrame(columns=['Covariance Type','Number of Cluster','Silhouette Score','Davies Bouldin Score'])
for i in parameters:
    for j in n_clusters:
        gmm_cluster=GaussianMixture(n_components=j,covariance_type=i,random_state=123)
        clusters=gmm_cluster.fit_predict(df_scaled)
        if len(np.unique(clusters))>=2:
           results_=results_.append({
           "Covariance Type":i,'Number ofCluster':j,
           "Silhouette Score":metrics.silhouette_score(df_scaled,clusters),
           'Davies Bouldin Score':metrics.davies_bouldin_score(df_scaled,clusters)}
           ,ignore_index=True)



gmm = GaussianMixture(n_components=5,covariance_type="diag",random_state=123)
gmm_cluster = gmm.fit_predict(df_scaled)



#algoritma karşılaştırma

from sklearn.metrics import silhouette_score



score_dbscn = silhouette_score(X, dbscn.labels_)
score_kmeans = silhouette_score(X, kmeans.labels_)
score_gmm = silhouette_score(X, gmm_cluster)

print("Silhouette score dbscn: " ,score_dbscn)
print("Silhouette score kmean: " ,score_kmeans)
print("Silhouette score gmm: ", score_gmm)


from sklearn.metrics import accuracy_score

accuracy_kmean = accuracy_score(y, kmeans.labels_)

print("Doğruluk skoru k mean:", accuracy_kmean)

accuracy_dbscn = accuracy_score(y, dbscn.labels_)

print("Doğruluk skoru dbscn:", accuracy_dbscn)

accuracy_gmm = accuracy_score(y, gmm_cluster)

print("Doğruluk skoru gmm:", accuracy_gmm)


from sklearn.metrics import davies_bouldin_score  #ideal olan 0

dav_kmean = davies_bouldin_score(X,kmeans.labels_)
print("Davies score k mean: ", dav_kmean)

dav_dbscn = davies_bouldin_score(X,dbscn.labels_)
print("Davies score dbscn: ", dav_dbscn)

dav_dbscn = davies_bouldin_score(X,gmm_cluster)
print("Davies score gmm: ",dav_dbscn)

####
