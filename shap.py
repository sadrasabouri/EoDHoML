import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import shap
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




"""## Select Features and target"""

r2_aggregated_TestScore =[]
r2_aggregated_TrainScore =[]
bunch_sample = np.arange(2000, len(data), 2000)
bunch_sample= np.append(bunch_sample,len(data))
bunch_sample

Group = ['A','B','C','D','E','F','G','H','I']
#Group = ["ABCDEFGHI"]
iterator= [0,1,2,3,4,5,6,7,8,9]
all_shap_values = []

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
data.rename(columns={'Depth.cm_Top': 'DT'}, inplace=True)

iterator = range(9)  # Assume iterator goes from 0 to 8
shap_df = pd.DataFrame()
shap_dfs_list = []

for i, (bunch, group) in zip(iterator, zip(bunch_sample, Group)):
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
    from sklearn.model_selection import RandomizedSearchCV

    
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

    explainer = shap.Explainer(best_regressor)
    shap_values = explainer(X_train)
    
    
    row = i // 3  # Calculate the row index
    col = i % 3   # Calculate the column index
    
    ax = axes[row, col]  # Get the corresponding subplot
    
    # Your existing code...
    
    explainer = shap.Explainer(best_regressor)
    shap_values = explainer(X_train)
 # Convert SHAP values to a numpy array and then to a DataFrame
# Convert SHAP values to a DataFrame and set the column name
   # Convert SHAP values to a DataFrame and set the column names
    shap_values_df = pd.DataFrame(shap_values.values, columns=[f'SHAP_{i+1}' for i in range(shap_values.shape[1])])
    
    # Rename columns in the SHAP values DataFrame based on X_train columns
    shap_values_df.columns = X_train.columns
    
    # Save the SHAP values DataFrame to a CSV file
    shap_values_df.to_csv(f'shap_values_iteration_{i}.csv', index=False)
    
    # Store the SHAP DataFrame in the list
    shap_dfs_list.append(shap_values_df)
    
    # Manually create SHAP summary plot and assign it to the subplot
    shap.summary_plot(shap_values, X_train, show=False)

    # Concatenate the SHAP values DataFrame with the main SHAP DataFrame

    # Concatenate the SHAP values DataFrame with the main SHAP DataFrame
    
    shap.summary_plot(shap_values, X_train, show=False)
    shap_df[f'SHAP_{group}'] = shap_values.values
    plt.sca(ax)  # Set the current axes to the subplot
    
    if row ==2 and col == 1:    
        ax.set_xlabel('SHAP Value',weight='bold')
    xtick_labels = ax.get_xticklabels()  # Add this line
    plt.setp(xtick_labels, weight='bold')  # Add this lin
    
    # Add text box with gray background and position it at left bottom of subplot
    
    props = dict(boxstyle='square', facecolor='white', alpha=0.1)
    ax.text(0.92, 0.05, f'{group}', transform=ax.transAxes,  fontweight='bold', bbox=props)
    
    props2 = dict(boxstyle='square', facecolor='white', alpha=0.1)
    
    #props2 = dict(boxstyle='square', facecolor='white', alpha=0.1)
    #ax.text(0.05, 0.92, f'R$^2$: {R2Train:.2f}\nRMSLE: {RMSLETrain:.2f}', transform=ax.transAxes, fontsize=FontSize, fontweight='bold', bbox=props2)
    # Set the y-axis label

    #ax.set_yticks(np.arange(0, 1.2, 0.2),weight='bold')
   # if  col == 0:
        #ax.set_ylabel('Estimated log($K_s$ [cm/hr])',weight='bold')
    ytick_labels = ax.get_yticklabels()  # Add this line
    plt.setp(ytick_labels, weight='bold')  # Add this lin
    #plt.savefig(r'C:/Users/nematirad/Desktop/reza/behzad/amin/shap finalll.svg', format="svg")

plt.tight_layout()

plt.show()
