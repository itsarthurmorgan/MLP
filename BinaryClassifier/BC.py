import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('adm_data.csv')

print("Column names in the dataset:")
print(data.columns)

data.drop('Serial No.', axis=1, inplace=True)

def handle_text_data(data):

    data.drop(['SOP', 'LOR'], axis=1, inplace=True)
    return data

data = handle_text_data(data.copy())  # Avoid modifying the original data

le = LabelEncoder()
data['University Rating'] = le.fit_transform(data['University Rating'])

scaler = StandardScaler()
data[['GRE', 'TOEFL Score', 'CGPA']] = scaler.fit_transform(data[['GRE', 'TOEFL Score', 'CGPA']])

print("Columns after preprocessing:")
print(data.columns)

if 'Research' in data.columns:
    target_column = 'Research'
else:
    raise KeyError("The target column 'Research' was not found in the dataset.")

X = data.drop(target_column, axis=1)
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

'''
Column names in the dataset:
Index(['Serial No.', 'GRE', 'TOEFL Score', 'University Rating', 'SOP', 'LOR',
       'CGPA', 'Research', 'admit '],
      dtype='object')
Columns after preprocessing:
Index(['GRE', 'TOEFL Score', 'University Rating', 'CGPA', 'Research',
       'admit '],
      dtype='object')
Accuracy: 0.7875
'''