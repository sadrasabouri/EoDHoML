# -*- coding: utf-8 -*-
"""Copy of Final_pachevsky_cleaned.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ne5YoLCQASW7bMAc6Qj0QEr9C_gyi5s9

# Applying Machine learning techniques (XGBoost) to predict of the Soil Hydrulic Conductivity

## Import libraries
"""
# f'$R^2$
import numpy as np
import pandas as pd
import sys
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib import rc, rcParams
import matplotlib.patches as mpl_patches
import math
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.utils import check_array
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import randint, uniform

# Define RMSLE as a custom scoring metric


"""## Read data"""
data     = pd.read_csv(r'C:\Users\nematirad\Desktop\reza\behzad\amin\USKSAT_OpenRefined_Cleaned.csv')

accuracy =  pd.read_csv(r'C:\Users\nematirad\Desktop\reza\behzad\amin\Final results with 10 itteration last.xlsx')

R2Test = accuracy['Test_r2 mean']
RMSLETest = accuracy['Test_RMSLE mean']
R2Train = accuracy['Train_r2 mean']
RMSLETrain = accuracy['Train RMSLE mean']


def isnotequal (x,y):
  if abs(x-y)<10**-2:
    return False
  return True

invalidindecies = np.array(list(map(isnotequal, data.iloc[: , 11:16].sum(axis=1), data.iloc[: , 16])))
data = data[~invalidindecies]
data.columns

"""## Drop rows with nan cell"""

data = data.dropna(subset=['Ksat_cmhr'])
data = data.dropna(subset=['Db'])
data = data.dropna(subset=['OC'])
data = data.dropna(subset=['Clay'])
data = data.dropna(subset=['Silt'])
data = data.dropna(subset=['Sand'])
data = data.dropna(subset=['VCOS'])
data = data.dropna(subset=['COS'])
data = data.dropna(subset=['MS'])
data = data.dropna(subset=['FS'])
data = data.dropna(subset=['VFS'])
data = data.dropna(subset=['Depth.cm_Top'])
data = data.dropna(subset=['Depth.cm_Bottom'])
data = data.dropna(subset=['Dia.cm'])
data = data.dropna(subset=['Height.cm'])
data

"""## Convert to Numeric"""

data['Silt'] = pd.to_numeric(data['Silt'], errors='coerce')
data['Clay'] = pd.to_numeric(data['Clay'], errors='coerce')
data['Sand'] = pd.to_numeric(data['Sand'], errors='coerce')
data['Depth.cm_Top'] = pd.to_numeric(data['Depth.cm_Top'], errors='coerce')
data['Depth.cm_Bottom'] = pd.to_numeric(data['Depth.cm_Bottom'], errors='coerce')
data['Height.cm'] = pd.to_numeric(data['Height.cm'], errors='coerce')
data['Dia.cm'] = pd.to_numeric(data['Dia.cm'], errors='coerce')
data['Db'] = pd.to_numeric(data['Db'], errors='coerce')
data['VCOS'] = pd.to_numeric(data['VCOS'], errors='coerce')
data['COS'] = pd.to_numeric(data['COS'], errors='coerce')
data['MS'] = pd.to_numeric(data['MS'], errors='coerce')
data['FS'] = pd.to_numeric(data['FS'], errors='coerce')
data['VFS'] = pd.to_numeric(data['VFS'], errors='coerce')
data['Ksat_cmhr'] = pd.to_numeric(data['Ksat_cmhr'], errors='coerce')

"""## Correlation numeric show"""


# Calculate the Pearson correlation coefficients for all variables



data = data.drop(columns=['Ref',
                          'Site',
                          'Soil',
                          'Sand',
                          'Field',
                          'Method',
                          'Depth.cm_Bottom',
                          'Dia.cm',
                          'Height.cm'])

"""## Check list of data for possible Nan cells"""

is_NaN = data.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = data[row_has_NaN]

print(rows_with_NaN)

"""## Distributon of ksat columns

"""



"""## Remove Outlier"""

highest_ksat_cmhr = data['Ksat_cmhr'].mean() + 2 * data['Ksat_cmhr'].std()
lowest_ksat_cmhr = data['Ksat_cmhr'].mean() - 2 * data['Ksat_cmhr'].std()



"""## Finding the Outliers"""

len(data[(data['Ksat_cmhr'] > highest_ksat_cmhr) | \
         (data['Ksat_cmhr'] < lowest_ksat_cmhr)])

"""## Trimming of Outliers"""

data = data[(data['Ksat_cmhr'] < highest_ksat_cmhr) & \
            (data['Ksat_cmhr'] > lowest_ksat_cmhr)]
data

"""## Apply the Capping"""

data['Ksat_cmhr'] = np.where(
    data['Ksat_cmhr'] > highest_ksat_cmhr,
    highest_ksat_cmhr,
    np.where(
        data['Ksat_cmhr'] < lowest_ksat_cmhr,
        lowest_ksat_cmhr,
        data['Ksat_cmhr']
    )
)



highest_ksat_cmhr = data['Ksat_cmhr'].mean() + 3 * data['Ksat_cmhr'].std()
lowest_ksat_cmhr = data['Ksat_cmhr'].mean() - 3 * data['Ksat_cmhr'].std()



data[(data['Ksat_cmhr'] > highest_ksat_cmhr) | \
     (data['Ksat_cmhr'] < lowest_ksat_cmhr)]
data = data[(data['Ksat_cmhr'] < highest_ksat_cmhr) & \
            (data['Ksat_cmhr'] > lowest_ksat_cmhr)]
data['Ksat_cmhr'] = np.where(
    data['Ksat_cmhr'] > highest_ksat_cmhr,
    highest_ksat_cmhr,
    np.where(
        data['Ksat_cmhr'] < lowest_ksat_cmhr,
        lowest_ksat_cmhr,
        data['Ksat_cmhr']
    )
)


"""## Select Features and target"""



r2_aggregated_TestScore =[]
r2_aggregated_TrainScore =[]
bunch_sample = np.arange(2000, 17646, 2000)
bunch_sample= np.append(bunch_sample,len(data))
bunch_sample

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
Group = ['A','B','C','D','E','F','G','H','I']
#Group = ["ABCDEFGHI"]
iterator= [0,1,2,3,4,5,6,7,8,9]
for i, bunch, group in zip(iterator,bunch_sample,Group):
    X = data.loc[:, data.columns != 'Ksat_cmhr']
    y = data.loc[:, data.columns == 'Ksat_cmhr']
# Randomly pick bunch of data by 2000 step
    slection = bunch / len(X)
    if slection != 1:
        X, X_test, y, y_test = train_test_split(X,y,test_size=1- slection,random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

    else:
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
    
    
    """## Feature Scaling"""
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train.values[:, :] = sc.fit_transform(X_train.values[:, :])
    X_test.values[:, :] =  sc.transform(X_test.values[:, :])
    
    
    
    
    """## XGBoost"""
    
    # model tuning
    
    from xgboost import XGBRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    import time
    
    # A parameter grid for XGBoost
    params = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 5),
        'min_child_weight': randint(1, 10),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)}
    
    reg = XGBRegressor(objective='reg:squarederror', nthread=4)
    
    # run randomized search
   # run randomized search
    n_iter_search = 100
    random_search = RandomizedSearchCV(reg,
                                      param_distributions=params,
                                      n_iter=n_iter_search,
                                      cv=5,
                                      scoring='neg_mean_squared_error')
    
    start = time.time()
    random_search.fit(X_train, y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))
    
    best_regressor = random_search.best_estimator_
    random_search.best_params_
    
    """## Get predictions"""
    
    y_pred_test = best_regressor.predict(X_test)
    
    y_pred_train = best_regressor.predict(X_train)
    
    """## Calculate MAE"""
    
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_squared_log_error
    
    rmse_pred = mean_absolute_error(y_test, y_pred_test) 
    

    
    
    
    print(f"Mean Squared Error = {mean_squared_error(y_test, y_pred_test)}")
    print("Root Mean Absolute Error = ", np.sqrt(rmse_pred))
    print(f"R-Squared = {r2_score(y_test, y_pred_test)}")
    print("Average log Residual =", np.mean(np.log(np.squeeze(y_pred_test)) - np.log(np.squeeze(y_test))))
    print(f"RMSLE = {np.mean((np.log(np.squeeze(1+ y_test)) - np.log(np.squeeze(1 + y_pred_test)))**2)**0.5}")
    
    """**Defining & calculating RMSLE&R2 for test and train dataset**"""
    
    RMSLEXGBoost_test = "{:.3f}".format(np.mean((np.log(np.squeeze(1+ y_test)) - np.log(np.squeeze(1 + y_pred_test)))**2)**0.5)
    RSquared_test = "{:.3f}".format(r2_score(y_test, y_pred_test))
    
    RMSLEXGBoost_train = "{:.3f}".format(np.mean((np.log(np.squeeze(1+ y_train)) - np.log(np.squeeze(1 + y_pred_train)))**2)**0.5)
    RSquared_train = "{:.3f}".format(r2_score(y_train, y_pred_train))
    
    RMSLEXGBoost_total = "{:.3f}".format(np.mean((np.log(np.squeeze(1+ np.concatenate((y_train,y_test)))) - np.log(np.squeeze(1 + np.concatenate((y_pred_train,y_pred_test)))))**2)**0.5)
    RSquared_total = "{:.3f}".format(r2_score(np.concatenate((y_train,y_test)), np.concatenate((y_pred_train,y_pred_test))))
    
    any(np.isnan(np.log(np.squeeze(1+ np.concatenate((y_train,y_test))))))
    
    np.sum(np.isnan(np.log(np.squeeze(1 + np.concatenate((y_pred_train,y_pred_test))))))
    
    # learning curve________________________________________________________________
    
    from sklearn.model_selection import learning_curve
    
    Len_X_train = len(X_train)
    train_sizes = list(range(1, int(Len_X_train *(1 - 0.2)) , 1000))
    
    
    """**Defining & calculating RMSLE&R2 for test and train dataset**"""
    from sklearn.metrics import make_scorer, mean_squared_log_error

# Define the RMSLE scorer
    

#_______________________________________________________________________
# Loop through the subplots and plot the data on a log-log scale

    
    row = i // 3
    col = i % 3
    font = {'family': 'Times New Roman',
            'size': 10}    
    ax = axs[row, col]
    plt.rcParams["font.family"] = "Times New Roman"
    FontSize = 12
    ax.scatter(np.log(y_test), np.log(y_pred_test), marker='o', s=2, c='b', label='Predicted')
    ax.scatter(np.log(y_test), np.log(y_test), marker='o', s=2, c='r', label='Actual')    # Add a diagonal line with slope 1 to each subplot
    line = (-2,5)
    ax.plot(line, line, 'r--', alpha=0.75, zorder=0)
    
    # Adjust the x and y axis limits between min and max of both
    
    ax.set_xlim([-2, 5])
    ax.set_ylim([-2, 5])

    if row ==2 and col == 1:    
        ax.set_xlabel('Measured log($K_s$ [cm/hr])',weight='bold',fontsize=FontSize)
    xtick_labels = ax.get_xticklabels()  # Add this line
    plt.setp(xtick_labels, weight='bold',fontsize=FontSize)  # Add this lin
    
    # Add text box with gray background and position it at left bottom of subplot
    
    props = dict(boxstyle='square', facecolor='white', alpha=0.1)
    ax.text(0.92, 0.05, f'{group}', transform=ax.transAxes, fontsize=FontSize, fontweight='bold', bbox=props)
    
    props2 = dict(boxstyle='square', facecolor='white', alpha=0.1)
    ax.text(0.05, 0.92, f'R$^2$: {R2Test:.2f}\nRMSLE: {RMSLETest:.2f}', transform=ax.transAxes, fontsize=FontSize, fontweight='bold', bbox=props2)
    
    #props2 = dict(boxstyle='square', facecolor='white', alpha=0.1)
    #ax.text(0.05, 0.92, f'R$^2$: {R2Train:.2f}\nRMSLE: {RMSLETrain:.2f}', transform=ax.transAxes, fontsize=FontSize, fontweight='bold', bbox=props2)
    # Set the y-axis label

    #ax.set_yticks(np.arange(0, 1.2, 0.2),weight='bold')
    if row ==1 and col == 0:
        ax.set_ylabel('Estimated log($K_s$ [cm/hr])',weight='bold',fontsize=FontSize)
    ytick_labels = ax.get_yticklabels()  # Add this line
    plt.setp(ytick_labels, weight='bold',fontsize=FontSize)  # Add this lin





    #plt.savefig(f'C:/Users/nematirad/Desktop/reza/behzad/amin/Erro bar {bunch}.svg',format='svg')
plt.tight_layout()
#plt.show()
