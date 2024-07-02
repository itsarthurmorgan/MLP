import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = pd.read_csv('adm_data.csv')  # Replace with your dataset file

print("Columns in the dataset before cleaning:", data.columns)

data.columns = data.columns.str.strip()
print("Columns in the dataset after cleaning:", data.columns)

data['admit'] = (data['admit'] >= 0.5).astype(int)

X = data[['GRE', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']]
y = data['admit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)  # Increase max_iter if necessary
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
