##################################################################################################################################################################
# PROJECT: EUREKA - Outlier Detection
# SECTION: Data Preprocessing Class
# AUTHOR: Dong-Hyuk Yang
# DATE: since 21.10.12
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

import config
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from scipy.stats import skew, kurtosis
# from saxpy.paa import paa
# from saxpy.sax import ts_to_string
# from saxpy.alphabet import cuts_for_asize #alphabet size
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.approximation import SymbolicAggregateApproximation

##################################################################################################################################################################
# set data preprocess class
##################################################################################################################################################################

class DataPrep(object):

    def __init__(self):
        # data configuration
        self.data_path      = config.data_path
        self.data_name_list = config.data_name_list

        self.drop_cols      = config.drop_cols
        self.pk_cols        = config.pk_cols

        self.rep_val_cols   = config.rep_val_cols
        self.rep_vals       = config.rep_vals

        self.norm_cols      = config.norm_cols
        self.corr_cols      = config.corr_cols

        self.paa_cols       = config.paa_cols
        self.sax_cols       = config.paa_cols

        # run load_data function
        # self.load_data()

    # load and merge sales
    def load_data(self) -> pd.DataFrame:
        # initialize df_list
        df_list = []

        # load each data for each file
        for df_name in self.data_name_list:
            df_tmp = pd.read_csv(df_name, index_col = None, header = 0)
            df_list.append(df_tmp)
        # end for

        # merge all df_list into a single dataframe
        df = pd.concat(df_list, axis = 0, ignore_index = True)

        return df

    # extract representative values
    def extract_rep_val(self, df: pd.DataFrame, cols: str) -> pd.DataFrame:
        # initialize result dataframe
        df_rep_val = pd.DataFrame(index = self.rep_val_cols, columns = self.rep_vals)

        # extract each rep value for each column
        for i, col in enumerate(cols):
            min  = np.quantile(df[col], q = 0)
            q_1  = np.quantile(df[col], q = 0.25)
            avg  = np.mean(df[col])
            med  = np.quantile(df[col], q = 0.5)
            q_3  = np.quantile(df[col], q = 0.75)
            max  = np.quantile(df[col], q = 1)
            ske  = skew(df[col])
            kurt = kurtosis(df[col], fisher = True)

            # aggregate rep values into a list
            rep_val_list_tmp = [min, q_1, avg, med, q_3, max, ske, kurt]
            rep_val_list     = [round(rep_val, 1) for rep_val in rep_val_list_tmp] # round value

            # bind rep values list into result dataframe
            df_rep_val.iloc[i, :] = rep_val_list
        # end for

        return df_rep_val

    # split dataset for each group
    def split_data(self, df: pd.DataFrame, cols: str) -> pd.DataFrame:
        # set group
        gb = df.groupby(cols)

        # aggregate each dataframe group
        df_split = [gb.get_group(x) for x in gb.groups]

        # sort data by SerialNo column for each dataframe
        for i in range(len(df_split)):
            df_split[i].sort_values('SerialNo')
        # end for

        return df_split

    # normalize data
    def normalize_date(self, df: pd.DataFrame, cols: str, n_quant: int, distribution: str) -> pd.DataFrame:
        # initialize result list that contains each dataframe group
        df_scaled = []

        # apply scaling for each data frame
        for i in range(len(df)):
            # deepcopy df and append into df_scaled
            df_scaled.append(df[i].copy(deep = True))

            # set QuantileTransformer scaler
            scaler = QuantileTransformer(n_quantiles = n_quant, output_distribution = distribution)

            # insert normalized values
            df_scaled[i].loc[:, cols] = scaler.fit_transform(df[i].loc[:, cols])
        # end for

        return df_scaled

    def calc_corr(self, df: pd.DataFrame, cols: str, method: str) -> pd.DataFrame:
        # initialize result list that contains correlation of each dataframe group
        df_corr = []

        # calculate correlation for each dataframe
        for i in range(len(df)):
            df_tmp  = df[i].loc[:, cols]
            corr    = df_tmp.corr(method = method)
            df_corr.append(corr)
        # end for

        return df_corr

    # apply piecewise aggregate approximation(paa)
    def paa(self, df: pd.DataFrame, cols: str, time_seg_size: int):
        # set paa model
        paa = PiecewiseAggregateApproximation(window_size = time_seg_size)

        # initialize result list that contains the result of paa
        df_paa = []

        # apply paa for each dataframe
        for i in range(len(df)):
            df_tmp    = df[i].loc[:, cols]
            df_tmp    = pd.concat([pd.Series(x) for x in df_tmp], axis = 1, keys = cols)
            df_tmp    = df_tmp.iloc[1:, :] # array type으로 변형해야 함!
            print(df_tmp.head())

            df_to_paa = paa.transform(df_tmp)
            df_paa.append(df_to_paa)
        # end for

        return df_paa

    # apply symbolic aggregation approximation(sax)
    def sax(self, df: pd.DataFrame, cols: str, time_segment_size: int, alphabet_size: int) -> pd.DataFrame:
        for i in range(len(df)):
            # set standard scaler
            scaler = StandardScaler()

            # insert normalized values
            df[i].loc[:, cols] = scaler.fit_transform(df[i].loc[:, cols])
        # end for

        return None
