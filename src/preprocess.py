import pandas as pd

def alter_features(X):

    return X


if __name__ == "__main__":
    # Read in our dataset
    data = pd.read_csv('data/train_data.csv')
    
    # We need to remove all rows that do not have a price

    # Save target values
    data.Price.to_csv('y_train.csv', index=False)
    data.drop('Price', axis=1, inplace=True)

    # Preprocess the features
    alter_features(data).to_csv('X_train.csv', index=False)





