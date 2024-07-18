import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset (replace with your dataset loading code)
# Assuming the dataset is stored in 'heart_data.csv' in the current directory
file_path = 'heartdisease.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset for inspection
print("Dataset preview:")
print(df.head())

# Separate features and target variable
X = df.drop('heartdisease', axis=1)  # Features
y = df['heartdisease']  # Target variable

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize PCA with desired number of components
pca = PCA(n_components=2)  # Reduce to 2 principal components for visualization

# Fit PCA on scaled data
X_pca = pca.fit_transform(X_scaled)

# Print explained variance ratio
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Print principal components (optional)
print("\nPrincipal Components:")
print(pca.components_)

# Print transformed data (first 5 rows)
print("\nTransformed Data (First 5 rows):")
print(X_pca[:5])


'''
Dataset preview:
   age  Gender  Family  diet  Lifestyle  cholestrol  heartdisease
0    0       0       1     1          3           0             1
1    0       1       1     1          3           0             1
2    1       0       0     0          2           1             1
3    4       0       1     1          3           2             0
4    3       1       1     0          0           2             0

Explained Variance Ratio:
[0.36443526 0.23327392]

Principal Components:
[[ 0.036049    0.47992002 -0.29409722 -0.53593346 -0.57998053  0.24141662]
 [ 0.58024685 -0.21085335  0.49551205 -0.09299285 -0.05336228  0.60152063]]

Transformed Data (First 5 rows):
[[-2.0939417  -1.12660249]
 [-1.13276948 -1.54889449]
 [ 0.41169176 -0.95554748]
 [-1.45319179  1.78114641]
 [ 1.93735453  1.28521989]]
'''