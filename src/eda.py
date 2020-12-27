import pandas as pd

X = pd.read_csv('data/training_features.csv', index_col=0)
y = pd.read_csv('data/training_target.csv', index_col=0)
new_X = pd.read_csv('data/new_properties.csv', index_col=0)

print('Features:')
print(X.columns)

print('\nTraining dataset:')
print(X.info())
print(X.head())

print('\nTarget dataset:')
print(y.info())
print(y.head())

print('\nNew dataset:')
print(new_X.info())
print(new_X.head())
