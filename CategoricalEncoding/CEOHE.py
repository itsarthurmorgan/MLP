import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample small dataset
data = {
    'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red'],
    'Size': ['S', 'M', 'L', 'M', 'S'],
    'Price': [10, 15, 20, 15, 10],
    'Class': ['A', 'B', 'A', 'A', 'B']
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Label Encoding
label_encoder = LabelEncoder()

# Apply Label Encoding to 'Color' column
df['Color_Label'] = label_encoder.fit_transform(df['Color'])
print("\nDataFrame after Label Encoding 'Color':")
print(df)

# One-hot Encoding using pandas get_dummies
df_one_hot = pd.get_dummies(df, columns=['Color', 'Size'])
print("\nDataFrame after One-hot Encoding with pd.get_dummies:")
print(df_one_hot)

# One-hot Encoding using sklearn's OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
# Fit and transform the data
one_hot_encoded = one_hot_encoder.fit_transform(df[['Color', 'Size']])

# Create a DataFrame with one-hot encoded columns
one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(['Color', 'Size']))
# Combine with the original DataFrame
df_combined = pd.concat([df, one_hot_df], axis=1)
print("\nDataFrame after One-hot Encoding with OneHotEncoder:")
print(df_combined)


'''
Original DataFrame:
   Color Size  Price Class
0    Red    S     10     A
1   Blue    M     15     B
2  Green    L     20     A
3   Blue    M     15     A
4    Red    S     10     B

DataFrame after Label Encoding 'Color':
   Color Size  Price Class  Color_Label
0    Red    S     10     A            2
1   Blue    M     15     B            0
2  Green    L     20     A            1
3   Blue    M     15     A            0
4    Red    S     10     B            2

DataFrame after One-hot Encoding with pd.get_dummies:
   Price Class  Color_Label  Color_Blue  Color_Green  Color_Red  Size_L  Size_M  Size_S
0     10     A            2       False        False       True   False   False    True
1     15     B            0        True        False      False   False    True   False
2     20     A            1       False         True      False    True   False   False
3     15     A            0        True        False      False   False    True   False
4     10     B            2       False        False       True   False   False    True
'''