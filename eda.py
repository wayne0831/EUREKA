##################################################################################################################################################################
# PROJECT: EUREKA - Outlier Detection
# SECTION: Exploratory Data Analysis
# AUTHOR: Dong-Hyuk Yang
# DATE: since 21.10.12
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

from DataPrep import DataPrep
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import warnings
import pickle

##################################################################################################################################################################
# set dataframe options
##################################################################################################################################################################

pd.set_option('display.width', 2000)           # show all columns of dataframe
pd.set_option('display.max_columns', None)     # show all columns of dataframe
pd.set_option('display.max_rows', None)        # show all rows of dataframe
# pd.set_option('mode.chained_assignment', None) # ignore warnings
warnings.filterwarnings(action = 'ignore')     # ignore user warnings

##################################################################################################################################################################
# load
##################################################################################################################################################################

# load class
prep = DataPrep()

# load pickle
with open('df_raw.pickle', 'rb') as f1:
    df_raw = pickle.load(f1)

with open('df.pickle', 'rb') as f2:
    df = pickle.load(f2)

with open('df_scaled.pickle', 'rb') as f3:
    df_scaled = pickle.load(f3)

##################################################################################################################################################################
# set specific data to be analyzed
##################################################################################################################################################################

# before/after scaling
idx           = 22
df_idx        = df[idx]
df_scaled_idx = df_scaled[idx]

print('*' * 200)
print('data: ', '\n')
print('raw data')
print(df_raw.head(), '\n')

print('df[', str(idx), ']')
print(df_idx.head(), '\n')

print('df_scaled[', str(idx), ']')
print(df_scaled_idx.head())
print('*' * 200)

##################################################################################################################################################################
# descriptive statistics
##################################################################################################################################################################

# before/after scaling
df_raw_rep_val        = prep.extract_rep_val(df = df_raw,        cols = prep.rep_val_cols)
df_idx_rep_val        = prep.extract_rep_val(df = df_idx,        cols = prep.rep_val_cols)
df_scaled_idx_rep_val = prep.extract_rep_val(df = df_scaled_idx, cols = prep.rep_val_cols)

print('*' * 200)
print('descriptive statistics: ', '\n')

print('raw data')
print(df_raw_rep_val, '\n')

print('df[', str(idx), ']')
print(df_idx_rep_val, '\n')

print('df_scaled[', str(idx), ']')
print(df_scaled_idx_rep_val)
print('*' * 200)

##################################################################################################################################################################
# plot histogram
##################################################################################################################################################################

# set font size
plt.rcParams.update({'font.size': 15})

# plot BLOSDC/RDI/FP in df_raw
plt.subplot(1, 3, 1)
plt.hist(df_raw['BLOSDC'], bins = 2000)
plt.xlim([-100, 100])
plt.title('BLOSDC')

plt.subplot(1, 3, 2)
plt.hist(df_raw['RDI'], bins = 2000)
plt.xlim([-5, 60])
plt.title('RDI')

plt.subplot(1, 3, 3)
plt.hist(df_raw['FP'], bins = 2000)
plt.xlim([1000, 7000])
plt.title('FP')
plt.show()

# plot BLOSDC/RDI/FP in df_idx
# plt.subplot(1, 3, 1)
# plt.hist(df_idx['RDI'], bins = 2000)
# plt.xlim([-5, 60])
# plt.show()
# plt.subplot(1, 3, 2)
# plt.hist(df_idx['RDI'], bins = 2000)
# plt.xlim([-5, 60])
# plt.show()
# plt.subplot(1, 3, 3)
# plt.hist(df_idx['FP'], bins = 2000)
# plt.xlim([-5, 60])
# plt.show()

##################################################################################################################################################################
# compute correlation
##################################################################################################################################################################

df_raw_corr = df_raw.loc[:, prep.corr_cols].corr(method = 'pearson')
df_corr     = prep.calc_corr(df = df, cols = prep.corr_cols, method = 'pearson') # pearson, spearman
df_idx_corr = df_corr[idx]

print('*' * 200)
print('correlation: ')
print('raw data')
print(df_raw_corr, '\n')

print('df[', str(idx), ']')
print(df_idx_corr)
# print(df_scaled_idx_corr)
print('*' * 200)

plt.scatter(df_idx['BLOSDC'], df_idx['RDI'])
# plt.scatter(df_scaled_idx['BLOSDC'], df_scaled_idx['RDI'])
plt.show()
