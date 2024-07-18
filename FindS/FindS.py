import csv

# Initialize the most specific hypothesis
hypo = ['%', '%', '%', '%', '%', '%']

# Read the CSV file
with open('trainingdata.csv') as csv_file:
    readcsv = csv.reader(csv_file, delimiter=',')
    data = [row for row in readcsv if row[-1].upper() == "YES"]

# Print the given training examples and positive examples
print("\nThe given training examples are:")
for row in data:
    print(row)

print("\nThe positive examples are:")
for row in data:
    print(row)
print("\n")

# Find-S Algorithm
print("The steps of the Find-S algorithm are:\n", hypo)
d = len(data[0]) - 1
hypo = data[0][:-1]

for instance in data:
    for j in range(d):
        if hypo[j] != instance[j]:
            hypo[j] = '?'
    print(hypo)

print("\nThe maximally specific Find-S hypothesis for the given training examples is:")
print(hypo)


'''
The given training examples are:
['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes']
['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes']
['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']

The positive examples are:
['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes']
['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes']
['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']


The steps of the Find-S algorithm are:
 ['%', '%', '%', '%', '%', '%']
['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same']
['Sunny', 'Warm', '?', 'Strong', 'Warm', 'Same']
['Sunny', 'Warm', '?', 'Strong', '?', '?']

The maximally specific Find-S hypothesis for the given training examples is:
['Sunny', 'Warm', '?', 'Strong', '?', '?']
'''