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
from scipy.stats import randint, uniform

#from google.colab import drive
#from google.colab import files
#drive.mount("/content/drive", force_remount=True)
#%cd /content/drive/My Drive/Pachevsky_sadra

"""## Read data"""
data     = pd.read_csv(r'C:\Users\nematirad\OneDrive - Kansas State University\Desktop\reza\behzad\amin\USKSAT_OpenRefined_Cleaned.csv')



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

highest_ksat_cmhr = data['Ksat_cmhr'].mean() + 3 * data['Ksat_cmhr'].std()
lowest_ksat_cmhr = data['Ksat_cmhr'].mean() - 3 * data['Ksat_cmhr'].std()



"""## Finding the Outliers"""

len(data[(data['Ksat_cmhr'] > highest_ksat_cmhr) | \
         (data['Ksat_cmhr'] < lowest_ksat_cmhr)])

"""## Trimming of Outliers"""

data = data[(data['Ksat_cmhr'] < highest_ksat_cmhr) & \
            (data['Ksat_cmhr'] > lowest_ksat_cmhr)]


"""## Select Features and target"""

param_df = pd.DataFrame(columns=['iteration', 'n_estimators', 'min_child_weight', 'gamma', 'learning_rate', 
                              'subsample', 'colsample_bytree', 'max_depth', 'objective', 'booster'])


r2_aggregated_TestScore =[]
r2_aggregated_TrainScore =[]
bunch_sample = np.arange(2000, len(data), 2000)
bunch_sample= np.append(bunch_sample,len(data))
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
Group = ['a','b','c','d','e','f','g','h','i']
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
        print(len(X_train))
    else:
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)
        print(len(X_train))

    
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

    best_params = random_search.best_params_
    param_df = param_df.append(best_params, ignore_index=True)
    best_regressor = random_search.best_estimator_
    random_search.best_params_
    
    """## Get predictions"""
    
    y_pred_test  = best_regressor.predict(X_test)
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
    
    
    
    """**Defining & calculating RMSLE&R2 for test and train dataset**"""
    #Len_X_train = len(X_train)
    #train_sizes = list(range(1, int(Len_X_train *(1 - 0.2)) , 1000))
    
    Len_X_train = len(X_train)
    train_sizes = np.linspace(5, Len_X_train, 10, dtype=int)
    
    #train_sizes, train_scores, test_scores = learning_curve(reg, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    # Calculate the learning curve
    train_sizes, train_scores, test_scores = learning_curve(
    best_regressor,  # the model that was found to be the best after tuning
    X_train,  # the training data
    y_train,  # the training labels
    cv=5,  # number of folds for cross-validation
    train_sizes=np.linspace(0.1, 1.0, 10),  # fractions of training set to use for learning curve
    scoring='r2',  # scoring metric
    n_jobs=-1  # use all available CPUs in parallel
    )
    
    '''   
    # Generate the learning curve
    train_sizes, train_scores, test_scores = learning_curve(
    reg,  # Your XGBoost model
    X_train, 
    y_train, 
    cv=5,  # Cross-validation parameter (can be set to a higher value if you prefer)
    train_sizes=train_sizes,  # Sizes for the learning curve
    scoring='r2',  # The scoring metric
    n_jobs=-1,  # Number of jobs to run in parallel
    shuffle=True  # Optional: Shuffle the training data before generating the curve
    )
    '''
    # Calculate the mean and standard deviation of the training and testing scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    r2_aggregated_TestScore.append(test_scores)
    r2_aggregated_TrainScore.append(train_scores)
    
  
 # Plot the learning curve in the appropriate subplot
    # plot the learning curve in the current subplot
    row = i // 3
    col = i % 3


    ax = axs[row, col]
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 12})
    # plot the train data with square markers and solid line
    ax.plot(train_sizes, train_mean, label='Train', color='maroon', marker='o' )
    # plot the test data with circle markers and dashed line
    ax.plot(train_sizes, test_mean, label='Cross-validation', color='darkblue', linestyle='--', marker='s')
    
    # connect the center of the circle markers with a dashed line

    ax.set_xlabel('')
    ax.set_ylabel('')
    # Set the x-axis labels
    if row == 2 and col == 1:
        ax.set_xlabel('Number of samples',weight='bold')
    # Set the x-axis tick labels to be bold
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontweight('bold')
    xtick_indices = list(train_sizes)
  
    # Set the number of x-ticks to 5 and show the x-axis numbers corresponding to each y-value
    ax.set_xticks(train_sizes[xtick_indices])
    ax.set_xticklabels(train_sizes[xtick_indices])
    
    #ax.set_title(f'{group}', weight='bold')  # position title in lower left corner of subplot
    # Add text box with gray background and position it at left bottom of subplot
    
    props = dict(boxstyle='square', facecolor='white', alpha=0.1)
    ax.text(0.05, 0.05, f'{group}', transform=ax.transAxes, fontweight='bold', bbox=props)
    # Set the y-axis label

    ax.set_yticks(np.arange(0, 1.05, 0.2))
    if row == 1 and col == 0:
        ax.set_ylabel('R$^2$',weight='bold')
    ytick_labels = ax.get_yticklabels()  # Add this line
    plt.setp(ytick_labels, weight='bold')  # Add this lin
    ax.set_ylim(0, 1.05)  # set y-axis limits
    #ax.tick_params(axis='y', which='both', labelweight='bold')
   

    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0),prop={'weight':'bold'})
    # Set the plot title

    

    fig.savefig(r'C:\Users\nematirad\OneDrive - Kansas State University\Desktop\reza\behzad\amin\learning_curve_final colorful modified test.svg', format='svg')


#train_sizes.append(int(Len_X_train *(1 - 0.2)))
plt.show()


