import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load the dataset
# Assuming the dataset is stored in 'heart_data.csv' in the current directory
file_path = 'heartdisease.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset for inspection
print("Dataset preview:")
print(df.head())

# Step 2: Prepare the data for modeling
X = df.drop('heartdisease', axis=1)  # Features
y = df['heartdisease']  # Target variable

# Step 3: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Standardize features (not necessary for SVM with linear kernel, but useful for PCA)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Perform PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to 2 principal components for visualization
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Step 6: Train SVM model with PCA-transformed data
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_pca, y_train)

# Step 7: Predict on test set and evaluate the model
y_pred = svm_model.predict(X_test_pca)

# Step 8: Calculate and print accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of SVM with PCA: {accuracy:.2f}")

# Step 9: Print classification report for detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


'''
Dataset preview:
   age  Gender  Family  diet  Lifestyle  cholestrol  heartdisease
0    0       0       1     1          3           0             1
1    0       1       1     1          3           0             1
2    1       0       0     0          2           1             1
3    4       0       1     1          3           2             0
4    3       1       1     0          0           2             0

Accuracy of SVM with PCA: 1.00

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00         1
           1       1.00      1.00      1.00         3

    accuracy                           1.00         4
   macro avg       1.00      1.00      1.00         4
weighted avg       1.00      1.00      1.00         4
'''