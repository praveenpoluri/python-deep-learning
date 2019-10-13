import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")
#Loading dataset
dataset = pd.read_csv('iris.csv')
print(dataset.columns)
# splitting the features and class
x = dataset.iloc[:,:6]
print(x.shape)
## Cleaning the data
nulls = pd.DataFrame(x.isnull().sum().sort_values(ascending=False)[:16])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)
x = x.select_dtypes(include=[np.number]).interpolate().dropna()
# Standard scaling the features
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)
# Building the k-means algorithm
from sklearn.cluster import KMeans
nclusters = 3 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(X_scaled)
# predict the cluster for each data point
y_cluster_kmeans = km.predict(X_scaled)
from sklearn import metrics
score = metrics.silhouette_score(X_scaled, y_cluster_kmeans)
print("Silhoutte Score : ", str(score))
#elbow method to know the number of clusters
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

# PCA
#from sklearn.decomposition import PCA
#pca = PCA(2)

#x_pca = pca.fit_transform(X_scaled)

#df2 = pd.DataFrame(data=x_pca)

#nclusters = 3 # this is the k in kmeans

#km = KMeans(n_clusters=nclusters)

#km.fit(df2)



# predict the cluster for each data point after applying PCA.

#y_cluster_kmeans = km.predict(df2)

#score = metrics.silhouette_score(df2, y_cluster_kmeans)

#print("Silhoutte Score with PCA: " + str(score))