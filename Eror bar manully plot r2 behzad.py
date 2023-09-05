import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

data     = pd.read_excel(r'C:\Users\nematirad\OneDrive - Kansas State University\Desktop\reza\behzad\amin\Final results with 10 itteration last.xlsx')

sample_size =data['sample size'] * 0.8
# Select the relevant columns from your dataframe
train_mean = data['Train_r2 mean']
train_std = data['Train_r2 std']
test_mean = data['Test_r2 mean']
test_std = data['Test_r2 std']

# Create a new figure
#plt.rcParams["font.family"] = "Times New Roman"
Font = 10
fig, ax = plt.subplots()
# Plot the train data with square markers and solid line


# Plot the test data with circle markers and dashed line
ax.errorbar(sample_size, train_mean, yerr=train_std, fmt='-s',markersize=3, color='maroon', label='Train', capsize=3)

# Plot the test data with circle markers and dashed line
ax.errorbar(sample_size, test_mean, yerr=test_std, fmt='-o',markersize=4, color='darkblue', label='Test', linestyle='--', capsize=3)


ax.legend(fontsize = Font,)
ax.set_ylabel('R$^2$', weight='bold',fontsize = Font)

# Set the background color to white
ax.set_facecolor('white')



plt.xticks(sample_size,fontsize = 8.5)
ax.set_ylim([0, 1.05])

ax.set_xlabel('Training sample size', weight='bold', fontsize = Font)
plt.subplots_adjust(bottom=0.15)

ytick_labels = ax.get_yticklabels()
plt.setp(ytick_labels, weight='bold',fontsize = 8.5)
xtick_labels = ax.get_xticklabels()
plt.setp(xtick_labels, weight='bold')

plt.savefig(r'C:/Users/nematirad/OneDrive - Kansas State University/Desktop/reza/behzad/amin/Erro bar final r2 behzad updated.svg', format='svg')
plt.show()



