import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('trainingdata.csv')
print(data)

# Separate features and target
concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])
print(concepts)
print(target)

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]
    
    for i, instance in enumerate(concepts):
        if target[i] == "Yes":
            for x in range(len(specific_h)):
                if instance[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        else:
            for x in range(len(specific_h)):
                if instance[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print(f"\nStep {i+1}")
        print("Specific:", specific_h)
        print("General:", general_h)
    
    general_h = [g for g in general_h if g != ['?' for _ in range(len(specific_h))]]
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print("\nFinal Specific Hypothesis:", s_final)
print("\nFinal General Hypothesis:", g_final)


'''
     sky airTemp humidity    wind water forecast enjoySport
0  Sunny    Warm   Normal  Strong  Warm     Same        Yes
1  Sunny    Warm     High  Strong  Warm     Same        Yes
2  Rainy    Cold     High  Strong  Warm   Change         No
3  Sunny    Warm     High  Strong  Cool   Change        Yes
[['Sunny' 'Warm' 'Normal' 'Strong' 'Warm' 'Same']
 ['Sunny' 'Warm' 'High' 'Strong' 'Warm' 'Same']
 ['Rainy' 'Cold' 'High' 'Strong' 'Warm' 'Change']
 ['Sunny' 'Warm' 'High' 'Strong' 'Cool' 'Change']]
['Yes' 'Yes' 'No' 'Yes']

Step 1
Specific: ['Sunny' 'Warm' 'Normal' 'Strong' 'Warm' 'Same']
General: [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]

Step 2
Specific: ['Sunny' 'Warm' '?' 'Strong' 'Warm' 'Same']
General: [['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]

Step 3
Specific: ['Sunny' 'Warm' '?' 'Strong' 'Warm' 'Same']
General: [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', 'Same']]

Step 4
Specific: ['Sunny' 'Warm' '?' 'Strong' '?' '?']
General: [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?'], ['?', '?', '?', '?', '?', '?']]

Final Specific Hypothesis: ['Sunny' 'Warm' '?' 'Strong' '?' '?']

Final General Hypothesis: [['Sunny', '?', '?', '?', '?', '?'], ['?', 'Warm', '?', '?', '?', '?']]
'''