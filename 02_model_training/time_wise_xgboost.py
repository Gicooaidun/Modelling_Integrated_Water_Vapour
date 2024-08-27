import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split,PredefinedSplit,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader,TensorDataset
from timeit import default_timer as timer 
import xgboost as xgb
from joblib import dump
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.metrics import mean_squared_error, mean_absolute_error
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# read data
train_set=pd.read_hdf('train_set.h5',header=0)
test_set=pd.read_hdf('test_set.h5',header=0)

# split the data into feature and target
X_train=train_set.drop('GPSIWV',axis=1)
y_train=train_set['GPSIWV']
X_test=test_set.drop('GPSIWV',axis=1)
y_test=test_set['GPSIWV']

# transform the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

month_to_fold = {
    1: 0, 2: 0,  # January and February as first fold
    3: 1, 4: 1,  # March and April as second fold
    5: 2, 6: 2,  # May and June as third fold
    7: 3, 8: 3,  # July and August as fourth fold
    9: 4, 10: 4  # September and October as fifth fold
}

train_set=pd.read_hdf('train_set.h5',header=0)
# Extracting month from the MultiIndex and mapping to fold numbers
fold_numbers = train_set.index.get_level_values(1).month.map(month_to_fold)

# Creating PredefinedSplit object
ps = PredefinedSplit(test_fold=fold_numbers)

param_distributions = {
    'max_depth': sp_randint(5, 20),
    'n_estimators': sp_randint(50, 200),
    'eta': uniform(0.05, 0.2),
    'subsample': uniform(0.5, 0.5)  
}
# Initialize the grid search model
grid_search = RandomizedSearchCV(estimator=xgb.XGBRegressor(),
                           param_distributions=param_distributions,
                           cv=ps,  # Use the PredefinedSplit
                           scoring='neg_mean_squared_error',
                           verbose=3,
                           n_iter=100)
# Perform grid search on the data
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

# Save the model
dump(grid_search, 'grid_search_results.joblib')
# Retrieve the best model from grid search
best_model = grid_search.best_estimator_


# Predict and calculate errors on the training set
y_train_pred = best_model.predict(X_train)

# Predict and calculate errors on the validation setS
y_test_pred = best_model.predict(X_test)

# Save the training and validation errors with indices to h5 files
train_errors_df = pd.DataFrame({'y_train': y_train,'y_train_pred':y_train_pred})
train_errors_df.index=train_set.index
train_errors_df.to_hdf('train_errors.h5', key='train_errors', mode='w')

test_errors_df = pd.DataFrame({'y_test':y_test,'y_test_pred':y_test_pred})
test_errors_df.index=test_set.index
test_errors_df.to_hdf('test_errors.h5', key='test_errors', mode='w')

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"mse_train:{mse_train}")
print(f"mse_test:{mse_test}")
print(f"mae_train:{mae_train}")
print(f"mae_test:{mae_test}")