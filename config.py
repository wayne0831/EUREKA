##################################################################################################################################################################
# PROJECT: EUREKA - Outlier Detection
# SECTION: Configuration
# AUTHOR: Dong-Hyuk Yang
# DATE: since 21.10.12
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

import pandas as pd
from glob import glob
import numpy as np

##################################################################################################################################################################
# set basic configuration
##################################################################################################################################################################

# identify data path/name list
data_path      = 'C:/Users/bimatrix/PycharmProjects/EUREKA/data'
data_name_list = glob(data_path + '/*.csv')

# set data preprocessing columns
pk_cols      = ['ComputerID', 'LotNumber']                                  # primary keys
drop_cols    = ['LotNo', 'WireNumber', 'IPAddress', 'Time', 'MeasureValue'] # useless columns

rep_vals     = ['Min', 'Q1', 'Avg', 'Med', 'Q3', 'Max', 'Skew', 'Kurt']     # rep values
rep_val_cols = ['BLOSDC', 'RDI', 'FP']  # columns to be applied in extracting rep values

norm_cols    = ['BLOSDC', 'RDI', 'FP']  # columns to be applied in normalization
corr_cols    = ['BLOSDC', 'RDI', 'FP']  # columns to be applied in correlation

paa_cols     = ['BLOSDC', 'RDI', 'FP']  # columns to be applied in paa
sax_cols     = ['BLOSDC', 'RDI', 'FP']  # columns to be applied in sax