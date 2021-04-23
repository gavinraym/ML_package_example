import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import explained_variance_score, max_error
import pickle



model = GradientBoostingRegressor()


holdout_X, train_X, holdout_y, train_y = (
    train_test_split(X, y, random_state=42,
    shuffle=True, stratify=y)
)

error = float() 
expl_var = float()

# In this for loop, we are testing our model after training on
# the training data only (NOT the holdout data). KFold splits
# the data into additional subsets to give us a more accurate
# read on our model's score.
for train_index, test_index in KFold(n_splits=5).split(train_X):
    model.fit(train_X.iloc[train_index], train_y.iloc[train_index])

    y_pred = model.predict(train_X.iloc[test_index])
    y_true = train_y.iloc[test_index]

    error += max_error(y_pred, y_true)
    expl_var += explained_variance_score(y_pred, y_true)

# After running KFold, we must devide by the number of folds to get
# an average for each score.
error /= 5
expl_var /= 5


# Here we are printing the testing scores of our model after using
# KFold. Before moving on, we will tune the model's hyper-parameters.
print(f'Training scores:\nACC = {accuracy}\nPRE = {precision}\nREC = {recall}')

# Once we have gained as much accuracy as possible from tuning the 
# model, we will score our model one final time using the holdout
# data we set aside at the beginning.
model.fit(train_X, train_y)
pred_y = model.predict(holdout_X)

accuracy = accuracy_score(holdout_y, pred_y)
precision = precision_score(holdout_y, pred_y)
recall = recall_score(holdout_y, pred_y)

# These are the most accurate scores we can get about our model.
print(f'Final Scores:\nACC = {accuracy}\nPRE = {precision}\nREC = {recall}')

# Even though it is necessary to leave the holdout data aside 
# when scoring, we can improve our model further by adding it
# back in for the final fitting.
model.fit(X, y)

# Saving the model is easy using the pickle module!
# load with [model = pickle.loads(filename)]
pickle.dump(model, open('BC_Model.pickle', 'wb'))
print('Model written to "BC_Model.pickle"')
