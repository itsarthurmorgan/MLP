import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import metrics

# Load the Heart Disease dataset (replace with your dataset loading code)
# Assuming your dataset is loaded into 'heart_disease.csv'
data = pd.read_csv('heart.csv')

# Data preprocessing
X = data.drop('target', axis=1)
y = data['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# EM Algorithm (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_clusters = gmm.fit_predict(X_scaled)

# k-Means Algorithm
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)

# Evaluate clustering performance (silhouette score)
gmm_silhouette = metrics.silhouette_score(X_scaled, gmm_clusters)
kmeans_silhouette = metrics.silhouette_score(X_scaled, kmeans_clusters)

# Print silhouette scores
print(f"GMM Silhouette Score: {gmm_silhouette:.2f}")
print(f"k-Means Silhouette Score: {kmeans_silhouette:.2f}")
