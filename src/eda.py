import pandas as pd

X = pd.read_csv('data/train_features.csv')
y = pd.read_csv('data/train_target.csv')
new_X = pd.read_csv('data/new_properties.csv')

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
