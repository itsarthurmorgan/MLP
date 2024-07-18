import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load the data
data = pd.read_csv("heartdisease.csv")
heart_disease = pd.DataFrame(data)
print(heart_disease)

# Define the Bayesian Network structure
model = BayesianNetwork([
    ('age', 'Lifestyle'),
    ('Gender', 'Lifestyle'),
    ('Family', 'heartdisease'),
    ('diet', 'cholestrol'),
    ('Lifestyle', 'diet'),
    ('cholestrol', 'heartdisease'),
    ('diet', 'cholestrol')
])

# Fit the model
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

# Perform inference
HeartDisease_infer = VariableElimination(model)

# Define valid input ranges
valid_ranges = {
    'age': (0, 4),
    'Gender': (0, 1),
    'Family': (0, 1),
    'diet': (0, 1),
    'Lifestyle': (0, 3),
    'cholestrol': (0, 2)
}

# Function to get validated input from the user
def get_valid_input(prompt, valid_range):
    while True:
        try:
            value = int(input(prompt))
            if value in range(valid_range[0], valid_range[1] + 1):
                return value
            else:
                print(f"Please enter a value between {valid_range[0]} and {valid_range[1]}")
        except ValueError:
            print("Invalid input. Please enter an integer.")

# Get inputs from the user
age = get_valid_input('Enter age (0-4): ', valid_ranges['age'])
gender = get_valid_input('Enter Gender (0 for Male, 1 for Female): ', valid_ranges['Gender'])
family = get_valid_input('Enter Family history (0 for No, 1 for Yes): ', valid_ranges['Family'])
diet = get_valid_input('Enter diet (0 for High, 1 for Medium): ', valid_ranges['diet'])
lifestyle = get_valid_input('Enter Lifestyle (0 for Athlete, 1 for Active, 2 for Moderate, 3 for Sedentary): ', valid_ranges['Lifestyle'])
cholestrol = get_valid_input('Enter cholestrol (0 for High, 1 for BorderLine, 2 for Normal): ', valid_ranges['cholestrol'])

# Query the model
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={
    'age': age,
    'Gender': gender,
    'Family': family,
    'diet': diet,
    'Lifestyle': lifestyle,
    'cholestrol': cholestrol
})

# Print the result
print(q)

'''
    age  Gender  Family  diet  Lifestyle  cholestrol  heartdisease
0     0       0       1     1          3           0             1
1     0       1       1     1          3           0             1
2     1       0       0     0          2           1             1
3     4       0       1     1          3           2             0
4     3       1       1     0          0           2             0
5     2       0       1     1          1           0             1
6     4       0       1     0          2           0             1
7     0       0       1     1          3           0             1
8     3       1       1     0          0           2             0
9     1       1       0     0          0           2             1
10    4       1       0     1          2           0             1
11    4       0       1     1          3           2             0
12    2       1       0     0          0           0             0
13    2       0       1     1          1           0             1
14    3       1       1     0          0           1             0
15    0       0       1     0          0           2             1
16    1       1       0     1          2           1             1
17    3       1       1     1          0           1             0
18    4       0       1     1          3           2             0
Enter age (0-4): 1
Enter Gender (0 for Male, 1 for Female): 1
Enter Family history (0 for No, 1 for Yes): 1
Enter diet (0 for High, 1 for Medium): 1
Enter Lifestyle (0 for Athlete, 1 for Active, 2 for Moderate, 3 for Sedentary): 1
Enter cholestrol (0 for High, 1 for BorderLine, 2 for Normal): 1
+-----------------+---------------------+
| heartdisease    |   phi(heartdisease) |
+=================+=====================+
| heartdisease(0) |              1.0000 |
+-----------------+---------------------+
| heartdisease(1) |              0.0000 |
+-----------------+---------------------+
'''