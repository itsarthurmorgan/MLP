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
