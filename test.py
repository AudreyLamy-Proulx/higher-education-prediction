import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load numeric data
df = pd.read_csv('students.csv')
X = df.select_dtypes(include=[np.number])

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=4, random_state=42).fit(X_pca)
ypred = kmeans.predict(X_pca)

print(ypred[:5])
