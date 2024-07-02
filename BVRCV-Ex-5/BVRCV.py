import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the dataset (assuming the dataset is named 'housing_data.csv')
data = pd.read_csv('housing.csv')

# Print column names for verification (optional)
print(data.columns)

# Remove duplicates if any
data.drop_duplicates(inplace=True)

# Convert 'medv' to a binary target: 1 if above median, 0 otherwise
median_medv = data['medv'].median()
data['medv_binary'] = (data['medv'] > median_medv).astype(int)

# Define features and target variable
X = data.drop(columns=['medv', 'medv_binary'])
y = data['medv_binary']

# Standardize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Bias and Variance
bias = np.mean(y_test - y_pred)
variance = np.var(y_pred)

print(f'Bias: {bias:.2f}')
print(f'Variance: {variance:.2f}')

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Perform K-fold cross-validation
def cross_validation(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    for train_index, test_index in kf.split(X):
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]
        
        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_test_cv)
        
        score = accuracy_score(y_test_cv, y_pred_cv)  # Use accuracy for classification
        fold_scores.append(score)
    
    avg_score = np.mean(fold_scores)
    return avg_score

# Compute cross-validation score
avg_accuracy = cross_validation(X, y, model)
print(f'Average Accuracy across 5 folds: {avg_accuracy:.2f}')
