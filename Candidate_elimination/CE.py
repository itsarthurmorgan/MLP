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
