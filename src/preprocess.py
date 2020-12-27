import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def alter_features(X):

    return X


if __name__ == "__main__":
    # Read in our dataset
    data = pd.read_csv('data/train_data.csv')
    
    # We need to remove all rows that do not have a price

    # Preprocess the features
    alter_features(data).to_csv('processed_data.csv', index=False)





