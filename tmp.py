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
idx           = 2
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

##################################################################################################################################################################
# apply paa
##################################################################################################################################################################

df_paa = prep.paa(df = df, cols = prep.paa_cols, time_seg_size = 10)

print(df_paa[idx])
