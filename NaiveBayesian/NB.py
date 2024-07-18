import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder

# Load the dataset from CSV
file_path = "tennisdata.csv"
df = pd.read_csv(file_path)

# Convert categorical variables to numerical labels
le = LabelEncoder()
df['Outlook'] = le.fit_transform(df['Outlook'])
df['Temperature'] = le.fit_transform(df['Temperature'])
df['Humidity'] = le.fit_transform(df['Humidity'])
df['Windy'] = le.fit_transform(df['Windy'])
df['PlayTennis'] = le.fit_transform(df['PlayTennis'])

# Define features and target variable
X = df[['Outlook', 'Temperature', 'Humidity', 'Windy']]
y = df['PlayTennis']

# Initialize the Naive Bayes classifier for categorical features
model = CategoricalNB()

# Train the model
model.fit(X, y)

# Make predictions (for demonstration, predict on the same data)
y_pred = model.predict(X)

# Print predictions and actual values
print("Predictions:", y_pred)
print("Actual Values:", y.values)

# Calculate accuracy (for demonstration, on training data)
accuracy = (y_pred == y.values).mean()
print(f"Accuracy: {accuracy:.2f}")


'''
Predictions: [0 0 1 1 1 1 1 0 1 1 1 1 1 0]
Actual Values: [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
Accuracy: 0.93
'''