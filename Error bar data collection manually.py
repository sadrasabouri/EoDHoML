import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")
from sklearn.model_selection import RandomizedSearchCV
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

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




"""## Select Features and target"""



train_RMSLE_mean_list =[]
train_RMSLE_std_list=[]
test_RMSLE_mean_list=[]
test_RMSLE_std_list=[]
train_r2_mean_list=[]
train_r2_std_list=[]
test_r2_mean_list=[]
test_r2_std_list=[]

Hyper_parameters =[]
bunch_sample = np.arange(2000, len(data), 2000)
bunch_sample= np.append(bunch_sample,len(data))
bunch_iteration =[]

#fig, axs = plt.subplots(3, 3, figsize=(12, 12))
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

    sc = StandardScaler()
    X.values[:, :] = sc.fit_transform(X.values[:, :])
    

   
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
    n_iter_search = 100
    random_search = RandomizedSearchCV(reg,
                                       param_distributions=params,
                                       n_iter=n_iter_search,
                                       cv=5,
                                       scoring='neg_mean_squared_error')
    
    
    start = time.time()
    train_RMSLE =[]
    test_RMSLE=[]
    train_r2=[]
    test_r2=[]

    param_df = pd.DataFrame(columns=['iteration', 'n_estimators', 'min_child_weight', 'gamma', 'learning_rate', 
                                 'subsample', 'colsample_bytree', 'max_depth', 'objective', 'booster'])

    for t in range(50):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state= t+1)
      
        # fit the randomized search to the data
        random_search.fit(X_train, y_train)
        # get the best hyperparameters and best model
        best_params = random_search.best_params_
        best_params['iteration'] = t+1
        param_df = param_df.append(best_params, ignore_index=True)
        
        best_regressor = random_search.best_estimator_
    
        # predict on test and train data using the best model
        y_pred_test = best_regressor.predict(X_test)
        y_pred_train = best_regressor.predict(X_train)
    
        
        RMSLE_test  = np.mean((np.log(np.squeeze(1+ y_test)) - np.log(np.squeeze(1 + y_pred_test)))**2)**0.5
        RMSLE_train = np.mean((np.log(np.squeeze(1+ y_train)) - np.log(np.squeeze(1 + y_pred_train)))**2)**0.5

        r2_test  = r2_score(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)

        
    

        train_RMSLE.append(RMSLE_train) 
        test_RMSLE.append(RMSLE_test)  
        
        train_r2.append(r2_train)
        test_r2.append(r2_test)
       
    
    # Calculate the mean and standard deviation of the training and testing scores
    train_RMSLE_mean = np.mean(train_RMSLE)
    train_RMSLE_std = np.std(train_RMSLE)
    
    test_RMSLE_mean = np.mean(test_RMSLE)
    test_RMSLE_std = np.std(test_RMSLE)
    
    train_r2_mean = np.mean(train_r2)
    train_r2_std = np.std(train_r2)
    
    test_r2_mean = np.mean(test_r2)
    test_r2_std = np.std(test_r2)
    
    Hyper_parameters.append(param_df)
    
    train_RMSLE_mean_list.append(train_RMSLE_mean)
    train_RMSLE_std_list.append(train_RMSLE_std)

    test_RMSLE_mean_list.append(test_RMSLE_mean)
    test_RMSLE_std_list.append(test_RMSLE_std)

    train_r2_mean_list.append(train_r2_mean)
    train_r2_std_list.append(train_r2_std)

    test_r2_mean_list.append(test_r2_mean)
    test_r2_std_list.append(test_r2_std)

    print(i)
    bunch_iteration.append(bunch)    
final_results = {'samplesize': bunch_iteration, 'Train RMSLE mean': train_RMSLE_mean_list,'Train RMSLE std': train_RMSLE_std_list,
        'Test_RMSLE mean': test_RMSLE_mean_list,'Test_RMSLE std':test_RMSLE_std_list,
        'Train_r2 mean': train_r2_mean_list, 'Train_r2 std': train_r2_std_list,
        'Test_r2 mean': test_r2_mean_list, 'Test_r2 std': test_r2_std_list,
        'Hyper parameters': Hyper_parameters}
df = pd.DataFrame.from_dict(final_results)

    
#_______________________________________________________________________
    
  