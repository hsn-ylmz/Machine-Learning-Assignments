# Hierarchical Clustering

# %reset -f

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# Using te dendrogram to find the optimal number of clusters
plt.figure(0)
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

# Fitting hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualing the clusters
plt.figure(1)
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0 ,1], s=100, c = 'red', edgecolors= 'black', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1 ,1], s=100, c = 'blue', edgecolors= 'black', label = 'Standart')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2 ,1], s=100, c = 'green', edgecolors= 'black', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3 ,1], s=100, c = 'cyan', edgecolors= 'black', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4 ,1], s=100, c = 'magenta', edgecolors= 'black', label = 'Sensible')
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()