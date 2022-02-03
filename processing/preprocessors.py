import numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import timedelta
import sys

DEBUG = False 

class DailyAggregator(BaseEstimator, TransformerMixin):
    ''' Aggregates time-series values to daily level.     '''
    def __init__(self, id_columns, time_column, value_columns ):
        super().__init__()
        if not isinstance(id_columns, list):
            self.id_columns = [id_columns]
        else:
            self.id_columns = id_columns

        self.time_column = time_column

        if not isinstance(value_columns, list):
            self.value_columns = [value_columns]
        else:
            self.value_columns = value_columns


    def fit(self, X, y=None): return self


    def transform(self, X):
        X = X.copy()
        X[self.time_column] = X[self.time_column].dt.normalize()
        X = X.groupby(by=self.id_columns + [self.time_column], as_index=False)[self.value_columns].sum()
        if DEBUG:
            print(f'-------after {__class__.__name__ }------------')
            print(X.head())
            print(X.shape)
        return X



class MissingTimeIntervalFiller(BaseEstimator, TransformerMixin):
    ''' Adds missing time intervals in a time-series dataframe.     '''
    DAYS = 'days'
    MINUTES = 'minutes'
    HOURS = 'hours'

    def __init__(self, id_columns, time_column, value_columns, time_unit, step_size ):
        super().__init__()
        if not isinstance(id_columns, list):
            self.id_columns = [id_columns]
        else:
            self.id_columns = id_columns

        self.time_column = time_column

        if not isinstance(value_columns, list):
            self.value_columns = [value_columns]
        else:
            self.value_columns = value_columns

        self.time_unit = time_unit
        self.step_size = int(step_size)

    
    def fit(self, X, y=None): return self # do nothing in fit
        

    def transform(self, X):
        min_time = X[self.time_column].min()
        max_time = X[self.time_column].max()      
        # print(min_time, max_time)  

        if self.time_unit == MissingTimeIntervalFiller.DAYS:
            num_steps = ( (max_time - min_time).days // self.step_size ) + 1
            all_time_ints = [min_time + timedelta(days=x*self.step_size) for x in range(num_steps)]

        elif self.time_unit == MissingTimeIntervalFiller.HOURS:
            time_diff_sec = (max_time - min_time).total_seconds()
            num_steps =  int(time_diff_sec // (3600 * self.step_size)) + 1
            num_steps = (max_time - min_time).days + 1
            all_time_ints = [min_time + timedelta(hours=x*self.step_size) for x in range(num_steps)]

        elif self.time_unit == MissingTimeIntervalFiller.MINUTES:
            time_diff_sec = (max_time - min_time).total_seconds()
            num_steps =  int(time_diff_sec // (60 * self.step_size)) + 1
            # print('num_steps', num_steps)
            all_time_ints = [min_time + timedelta(minutes=x*self.step_size) for x in range(num_steps)]
        else: 
            raise Exception(f"Unrecognized time unit: {self.time_unit}. Must be one of ['days', 'hours', 'minutes'].")

        # create df of all time intervals
        full_intervals_df = pd.DataFrame(data = all_time_ints, columns = [self.time_column])   

        # get unique id-var values from original input data
        id_cols_df = X[self.id_columns].drop_duplicates()
        
        # get cross join of all time intervals and ids columns
        full_df = id_cols_df.assign(foo=1).merge(full_intervals_df.assign(foo=1)).drop('foo', 1)

        # merge original data on to this full table
        full_df = full_df.merge(X[self.id_columns + [self.time_column] + self.value_columns], 
        on=self.id_columns + [self.time_column], how='left')
        if DEBUG:
            print(f'-------after {__class__.__name__ }------------')
            print(full_df.head())
            print(full_df.shape)
        return full_df



class DataPivoter(BaseEstimator, TransformerMixin):
    '''  Pivots a dataframe with a given column '''

    def __init__(self, non_pivoted_columns, pivoting_column, pivoted_columns, fill_na_val):
        super().__init__()    
        self.non_pivoted_columns = \
            [non_pivoted_columns] if not isinstance(non_pivoted_columns, list) else non_pivoted_columns
        self.pivoted_columns = [pivoted_columns] if not isinstance(pivoted_columns, list) else pivoted_columns
        self.pivoting_column = pivoting_column
        self.fill_na_val = fill_na_val


    def fit(self, X, y=None): return self # do nothing in fit


    def transform(self, X):
        processed_X = X.pivot_table(index = self.non_pivoted_columns, 
                        aggfunc=sum,
                        columns=self.pivoting_column, 
                        values=self.pivoted_columns, 
                        fill_value = self.fill_na_val
                        ).reset_index()

        
        # pivot table will result in multi column index. To get a regular column names
        processed_X.columns = [ col[0] if col[1] == '' else col[1]  for col in processed_X.columns ]  
        if DEBUG:
            print(f'-------after {__class__.__name__ }------------')
            print(processed_X.head())
            print(processed_X.shape) 
        return processed_X

    
    def inverse_transform(self, preds_df):
        # unpivot given dataframe
        preds_df2 = pd.melt(preds_df.reset_index(), 
            id_vars=self.non_pivoted_columns,
            value_vars=preds_df.columns,
            var_name = self.pivoting_column,
            value_name = self.pivoted_columns[0]
            )
        return preds_df2



class IndexSetter(BaseEstimator, TransformerMixin):
    ''' Set index '''
    def __init__(self, index_cols, drop_existing):
        self.index_cols = index_cols
        self.drop_existing = drop_existing
    
    def fit(self, X, y=None): return self # do nothing in fit


    def transform(self, X):
        X = X.copy()
        X.reset_index(drop=self.drop_existing, inplace=True)
        X.set_index(self.index_cols, inplace=True)
        if DEBUG:
            print(f'-------after {__class__.__name__ }------------')
            print(X.head())
            print(X.shape) 
        return X



class SubTimeSeriesSampler(BaseEstimator, TransformerMixin):
    ''' Samples a sub-series of length t <= the original series of length T. Assumes series is in columns 
    Original time-series time labels (column headers) are replaced with t_0, t_1, ... t_<series_len>.
    '''
    def __init__(self, series_len, num_reps): 
        self.series_len = series_len
        self.num_reps = num_reps


    def fit(self, X, y=None): return self


    def transform(self, X):
        curr_len = X.shape[1]

        if curr_len < self.series_len: 
            raise Exception(f"Error sampling series. Target length {self.series_len} exceeds current length {curr_len}")

        sampled_data = []
        data_arr = X.values
        for _ in range(self.num_reps):
            for i in range(data_arr.shape[0]):
                rand_idx = np.random.randint(0, curr_len - self.series_len)
                sampled_data.append( data_arr[i, rand_idx: rand_idx + self.series_len] )
        
        idx = list(X.index) * self.num_reps
        col_names = [ f't_{i}' for i in range(self.series_len)]
        sampled_data = pd.DataFrame(sampled_data, columns=col_names, index= idx)
        if DEBUG:
            print(f'-------after {__class__.__name__ }------------')
            print(sampled_data.head())
            print(sampled_data.shape) 
        return sampled_data



class AddLeftRightFlipper(BaseEstimator, TransformerMixin):
    '''
    Adds left right flipped version of tensor
    '''
    def __init__(self): pass
    def fit(self, X, y=None): return self

    def transform(self, X):
        X_flipped = pd.DataFrame( np.fliplr(X), columns=X.columns, index=X.index )
        X = pd.concat([X, X_flipped], axis=0, ignore_index=True)
        if DEBUG:
            print(f'-------after {__class__.__name__ }------------')
            print(X.head())
            print(X.shape) 
        return X



class SeriesLengthTrimmer(BaseEstimator, TransformerMixin):
    '''
    Trims the length of a series to use latest data points 
    '''
    def __init__(self, series_len): 
        self.series_len = series_len

    def fit(self, X, y=None): return self

    def transform(self, X):
        curr_len = X.shape[1]

        if curr_len < self.series_len: 
            raise Exception(f"Error trimming series. Target length {self.series_len} exceeds current length {curr_len}")
        
        X_vals = X.values[:, -self.series_len:]
        col_names = [ f't_{i}' for i in range(self.series_len)]
        X_vals = pd.DataFrame(X_vals, columns=col_names, index=X.index)        
        if DEBUG:
            print(f'-------after {__class__.__name__ }------------')
            print(X_vals.head())
            print(X_vals.shape) 
        return X_vals



class DFShuffler(BaseEstimator, TransformerMixin):
    def __init__(self, shuffle = True): 
        self.shuffle = shuffle

    def fit(self, X, y=None): return self

    def transform(self, X, y=None): 
        if self.shuffle == False: return X  
        X = X.sample(frac=1)   
        if DEBUG:
            print(f'-------after {__class__.__name__ }------------')
            print(X.head())
            print(X.shape) 
        return X



class TSMinMaxScaler2(BaseEstimator, TransformerMixin):
    '''Scales history and forecast parts of time-series based on history data'''
    def __init__(self, scaling_len, upper_bound = 5.): 
        if scaling_len < 2: raise Exception("Min Max scaling length must be >= 2")
        self.scaling_len = scaling_len
        self.max_scaler = MinMaxScaler()
        self.row_sums = None
        self.upper_bound = upper_bound
        

    def fit(self, X, y=None):         
        return self
    
    def transform(self, X, y=None): 
        curr_len = X.shape[1]
        if curr_len < self.scaling_len: 
            msg = f''' Error scaling series. 
            Sum of scaling_len {self.scaling_len} should not exceed series length {curr_len}.  '''
            raise Exception(msg)
        
        df = X  if curr_len == self.scaling_len  else  X[ X.columns[ : self.scaling_len ] ] 
        self.row_sums = df.sum(axis=1)
        df = df[self.row_sums != 0]
        self.max_scaler.fit(df.T)
            
        # print(X.shape, self.row_sums.shape)
        # sys.exit()
        X_filtered = X[self.row_sums != 0].copy()
        vals = self.max_scaler.transform(X_filtered.T).T
        vals = np.where(vals > self.upper_bound, self.upper_bound, vals)

        X = pd.DataFrame(vals, columns=X_filtered.columns, index=X_filtered.index)
        if DEBUG:
            print(f'-------after {__class__.__name__ }------------')
            print(X.head())
            print(X.shape) 
        return X

    def inverse_transform(self, X):
        return self.max_scaler.inverse_transform(X.T).T



class TSMinMaxScaler(BaseEstimator, TransformerMixin):
    '''Scales history and forecast parts of time-series based on history data'''
    def __init__(self, scaling_len, upper_bound = 5.): 
        if scaling_len < 2: raise Exception("Min Max scaling length must be >= 2")
        self.scaling_len = scaling_len
        self.min_vals = None      
        self.max_vals = None  
        self.ranges = None  
        self.upper_bound = upper_bound
        

    def fit(self, X, y=None):  return self

    
    def transform(self, X, y=None): 

        if self.scaling_len < 1: 
            msg = f''' Error scaling series. 
            scaling_len needs to be at least 2. Given length is {self.scaling_len}.  '''
            raise Exception(msg)
        

        X_vals = X.values
        self.min_vals = np.expand_dims( X_vals[ :,  : self.scaling_len  ].min(axis=1), axis = 1)
        self.max_vals = np.expand_dims( X_vals[ :,  : self.scaling_len  ].max(axis=1), axis = 1)

        self.ranges = self.max_vals - self.min_vals
        self.ranges = np.where(self.ranges == 0, 1e-5, self.ranges)
        # print(self.min_vals.shape, self.ranges.shape)

        # sys.exit()
        X_vals = X_vals - self.min_vals
        X_vals = np.divide(X_vals, self.ranges)        
        X_vals = np.where( X_vals < self.upper_bound, X_vals, self.upper_bound)

        X = pd.DataFrame(X_vals, columns=X.columns, index=X.index)
        if DEBUG:
            print(f'-------after {__class__.__name__ }------------')
            print(X.head())
            print(X.shape) 
        return X
        

    def inverse_transform(self, X):
        X = X * self.ranges
        X = X + self.min_vals
        return X



class TimeSeriesXYSplitter(BaseEstimator, TransformerMixin):
    '''Splits the time series into X (history) and Y (forecast) series'''
    def __init__(self, X_len, Y_len): 
        self.X_len = X_len
        self.Y_len = Y_len
        

    def fit(self, X, y=None): return self

    def transform(self, X, y=None): 
        curr_len = X.shape[1]
        encode_len  = self.X_len
        decode_len = (0 if self.Y_len == 'auto' else self.Y_len)

        if curr_len < encode_len + decode_len: 
            msg = f''' Error splitting series. 
            Sum of X_len {self.X_len} and Y_len {self.Y_len} should not exceed series length {curr_len}.  '''
            raise Exception(msg)

        # bit of a hack but sklearn pipeline only allows one thing to be returned in transform()
        cols = X.columns 
        if self.Y_len == 'auto':  return { 'X': X[cols[-self.X_len :]], 'Y': X[cols[-self.X_len :]] }
        if self.Y_len == 0:  return { 'X': X[cols[-self.X_len :]], 'Y': pd.DataFrame() }
        return {
            'X': X[cols[-( self.X_len + self.Y_len) :  -self.Y_len] ], 
            'Y':X[cols[ -self.Y_len : ] ] 
            }



if __name__ == "__main__": 

    # data = pd.read_parquet("wfm_single_q_Internal_daily_history.parquet")
    # data = pd.read_parquet("WFM_200q_Internal_daily_history.parquet")
    # data.rename(columns={ 'queueid': 'seriesid', 'date': 'ts', 'callvolume': 'v',}, inplace=True)
    
    data = pd.read_parquet("History_series_0028C91B.002795_filled.parquet")
    data.rename(columns={ 'queueid': 'seriesid', 'time': 'ts', 'callvolume': 'v',}, inplace=True)

    
    data['ts'] = pd.to_datetime(data['ts'])
    data = data[['seriesid', 'ts', 'v']]

    hist_len = 365
    fcst_len = 90

    print("-----------orig data -------------------")
    # print(data.head()); print(data.shape)   
     
    print("-----------after daily agg -------------------")
    agg = DailyAggregator('seriesid', 'ts', 'v')
    data = agg.fit_transform(data)
    # print(data.head()); print(data.shape)    

    print("-----------after adding missing intervals -------------------")
    filler = MissingTimeIntervalFiller('seriesid', 'ts', 'v', 'days', 1)
    data = filler.fit_transform(data)
    # print(data.head()); print(data.shape) 

    print("-----------after pivoting -------------------")
    pivoter = DataPivoter('seriesid', 'v', 'ts', 0)
    data = pivoter.fit_transform(data)
    # print(data.head()); print(data.shape) 

    print("-----------after indexing -------------------")
    indexer = IndexSetter('seriesid', drop_existing=True)
    data = indexer.fit_transform(data)
    # print(data.head()); print(data.shape) 

    print("-----------after sampling -------------------")
    sampler = SubTimeSeriesSampler(series_len=hist_len+fcst_len, num_reps=5)
    data = sampler.fit_transform(data)
    # print(data.head()); print(data.shape) 

    print("-----------after shuffling -------------------")
    shuffler = DFShuffler()
    data = shuffler.fit_transform(data)
    print(data.head()); print(data.shape) 

    print("-----------after max scaling -------------------")
    scaler = TSMinMaxScaler(scaling_len=hist_len)
    data = scaler.fit_transform(data)
    print(data.head()); print(data.shape) 

    print("-----------after X Y split -------------------")
    splitter = TimeSeriesXYSplitter(hist_len, fcst_len)
    data = splitter.fit_transform(data)
    print(data.keys())
    print(data['X'])
    print(data['Y'])



