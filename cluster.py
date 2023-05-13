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

dbscn = DBSCAN(eps=3.227, min_samples=2,metric="euclidean") #değiştirince azaldı orangedan baktım
clusters = dbscn.fit_predict(df_scaled)

#print(clusters)

outliers_dbscn = np.where(clusters == -1)

plt.scatter(df_scaled[:, 2], df_scaled[:, 1], c="blue", alpha=0.5)
# Aykırı verileri ayrıca görselleştirme
plt.scatter(df_scaled[outliers_dbscn, 2], df_scaled[outliers_dbscn, 1], c='red', marker='x', alpha=1)
plt.xlabel("Comment Count")
plt.ylabel("Like Count")
plt.show()


print("Number of outliers:", len(outliers_dbscn))
print(outliers_dbscn)

#data_o = data.drop(outliers[0],axis = 0)

######################################################

# K-means kümeleme
kmeans = KMeans(n_clusters=2, random_state=1, n_init="auto").fit(df_scaled)
clusters_kmean = kmeans.fit_predict(df_scaled)


# Aykırı değerleri tespit etme
distances = kmeans.transform(df_scaled)
std_dev = np.std(distances, axis=0)
mean_dist = np.mean(distances, axis=0)

threshold = mean_dist + 2 * std_dev #Normal Distribution
outliers_kmean = np.where(np.any(distances > threshold, axis=1))[0]

plt.scatter(df_scaled[:, 2], df_scaled[:, 1], c="blue", alpha=0.5)
# Aykırı verileri ayrıca görselleştirme
plt.scatter(df_scaled[outliers_kmean, 2], df_scaled[outliers_kmean, 1], c='red', marker='x', alpha=1)
plt.xlabel("Comment Count")
plt.ylabel("Like Count")
plt.show()



print("Number of outliers:", len(outliers_kmean))
print(outliers_kmean)

######################################################
#GMM Cluster

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




gmm = GaussianMixture(n_components=2,covariance_type="spherical",random_state=123)
gmm_cluster = gmm.fit_predict(df_scaled)

#thresholz Z-score a göre yapılmıştır
outliers_gmm = np.where(threshold > 3)[0]

plt.scatter(df_scaled[:, 2], df_scaled[:, 1], c="blue", alpha=0.5)
plt.scatter(df_scaled[outliers_gmm, 2], df_scaled[outliers_gmm, 1], c='red', marker='x')
plt.xlabel("Comment Count")
plt.ylabel("Like Count")
plt.show()

print("Aykırı veri sayısı:", len(outliers_gmm))
print(outliers_gmm)


############################################################
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

