
from sklearn.pipeline import Pipeline
import pandas as pd
from datetime import datetime, timedelta
from processing import preprocessors as pp 
from config import config as cfg 


def get_preprocess_pipelines(encode_len, decode_len):

    # both train and predict
    pipeline1 = Pipeline(
        [
            # aggregate volumes at daily level
            (
                cfg.DAILY_AGGREGATOR,
                pp.DailyAggregator(
                    id_columns = cfg.ID_COL, 
                    time_column = cfg.TIME_COL, 
                    value_columns = cfg.VALUE_COL 
                    )
            ),
            # add missing time intervals
            (
                cfg.MISSING_TIME_INTS_FILLER,
                pp.MissingTimeIntervalFiller(
                    id_columns = cfg.ID_COL,
                    time_column = cfg.TIME_COL, 
                    value_columns = cfg.VALUE_COL,
                    time_unit = 'days', 
                    step_size = 1,
                    )
            ),
            # Pivot the time column into columns
            (
                cfg.TIME_PIVOTER,
                pp.DataPivoter(
                    non_pivoted_columns = cfg.ID_COL,
                    pivoting_column = cfg.TIME_COL, 
                    pivoted_columns = cfg.VALUE_COL, 
                    fill_na_val = 0.,
                    )
            ),
            # set index to series_id
            (
                cfg.INDEX_SETTER,
                pp.IndexSetter(
                    index_cols = cfg.ID_COL,
                    drop_existing = True, 
                    )
            )
        ]
    )

    # only in training
    pipeline2 = Pipeline(
        [
            # subsample from series
            (
                cfg.SERIES_SUBSAMPLER,
                pp.SubTimeSeriesSampler(
                    series_len = encode_len + (0 if decode_len == 'auto' else decode_len),
                    num_reps = cfg.NUM_REPS_PRETRAINING_DATA, 
                    )
            ),
            # do left right flip
            (
                cfg.LEFT_RIGHT_FLIPPER,
                pp.AddLeftRightFlipper()
            ),
            # shuffle data
            (
                cfg.SERIES_SHUFFLER,
                pp.DFShuffler(
                    shuffle = True
                    )
            ),
            # trim series to length 
            (
                cfg.SERIES_TRIMMER,
                pp.SeriesLengthTrimmer(
                    series_len = encode_len + (0 if decode_len == 'auto' else decode_len),
                    )
            ),
        ]
    )

    # predict only
    pipeline3 = Pipeline(
        [            
            # trim series to length 
            (
                cfg.SERIES_TRIMMER,
                pp.SeriesLengthTrimmer(
                    series_len = encode_len ,
                    )
            ),            
        ]
    )

    # both
    pipeline4 = Pipeline(
        [            
            # Min max scale data
            (
                cfg.MINMAX_SCALER,
                pp.TSMinMaxScaler(
                    scaling_len = encode_len,
                    upper_bound = cfg.MAX_SCALER_UPPER_BOUND
                    )
            ),            
        ]
    )

    # train only
    pipeline5 = Pipeline(
        [
            # Split into X and Y dataframes
            (
                cfg.XY_SPLITTER,
                pp.TimeSeriesXYSplitter(
                    X_len = encode_len,
                    Y_len = decode_len,
                    )
            )
        ]
    )

    # predict only
    pipeline6 = Pipeline(
        [
            # Split into X and Y dataframes
            (
                cfg.XY_SPLITTER,
                pp.TimeSeriesXYSplitter(
                    X_len = encode_len,
                    Y_len = 0,
                    )
            )
        ]
    )

    training_pipeline = Pipeline( pipeline1.steps + pipeline2.steps + pipeline4.steps + pipeline5.steps)
    prediction_pipeline = Pipeline( pipeline1.steps + pipeline3.steps + pipeline4.steps + pipeline6.steps)

    return training_pipeline, prediction_pipeline





if __name__ == '__main__':
	# ----------------------------------------------------------------------------------
	# get sample training data
	data = pd.read_parquet(cfg.TEST_FORECAST_FILE, columns=['queueid', 'date','callvolume'])
	data['date'] = pd.to_datetime(data['date'])   
	data.rename(columns={ 'queueid': 'seriesid', 'date': 'ts', 'callvolume': 'v',}, inplace=True)
	

	# our training data filtered to leave out recent history 
	train_data = data[data.ts <= datetime.strptime('11/30/2018', '%m/%d/%Y') ]


	decode_len = 84
	encode_len = 365*3 
	pipe1, pipe2 = get_preprocess_pipelines(
			with_train_steps = True,
			encode_len = encode_len,
			decode_len = decode_len,
			shuffle = True
			)
	data = pipe1.fit_transform(train_data)
	X = data['X']; Y = data['Y']
	print('pre-processed shape', X.shape, Y.shape)
