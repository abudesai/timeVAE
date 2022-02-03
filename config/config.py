
import os 

PWD = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(PWD, '../../..'))


DATASET_DIR = os.path.join(PACKAGE_ROOT, 'data/wfm')



# DATASET_NAME = 'synthetic_history_daily'
#DATASET_NAME = 'WFM_200q_Internal_daily_history'
DATASET_NAME = 'wfm_single_q_Internal_daily_history' 
# DATASET_NAME = 'Filtered_300to600_Internal_daily_history'


DATA_FILE_NAME = f'{DATASET_NAME}.parquet'
DATA_FILE_PATH_AND_NAME = os.path.join(DATASET_DIR, DATA_FILE_NAME)



MODEL_DIR = os.path.abspath(os.path.join(PWD, '..'))
TRAINED_MODEL_DIR = os.path.join(MODEL_DIR, 'trained_models')

TRAIN_PIPE_FILE = os.path.join(TRAINED_MODEL_DIR, 'sklearn_pipeline_train.pkl') 
PRED_PIPE_FILE = os.path.join(TRAINED_MODEL_DIR, 'sklearn_pipeline_pred.pkl') 
PARAMS_FILE = os.path.join(TRAINED_MODEL_DIR, 'parameters.pkl') 

ENCODER_WEIGHTS = os.path.join(TRAINED_MODEL_DIR, 'encoder_weights.h5') 
DECODER_WEIGHTS = os.path.join(TRAINED_MODEL_DIR, 'decoder_weights.h5') 

# ------------------------------------------------------

ID_COL = 'seriesid'
TIME_COL = 'ts'
VALUE_COL = 'v'

EXOG_VALUE_COL_PREFIX = 'exog_'


WEEKDAY_COL = f'{EXOG_VALUE_COL_PREFIX}dow'
DAY_IN_MONTH_COL = f'{EXOG_VALUE_COL_PREFIX}dom'
PERIOD_IN_MONTH_COL = f'{EXOG_VALUE_COL_PREFIX}pom'
WEEK_OF_YEAR_COL = f'{EXOG_VALUE_COL_PREFIX}wiy'
MONTH_OF_YEAR_COL = f'{EXOG_VALUE_COL_PREFIX}moy'
YEAR_COL = f'{EXOG_VALUE_COL_PREFIX}yr_'
DAY_NUM_COL = f'{EXOG_VALUE_COL_PREFIX}day_num_'
SP_DAY_COL = f'{EXOG_VALUE_COL_PREFIX}sp_day_'
DAY_NUM_SCALED_COL = 'scaled_day_num'
UNITY_COL = 'unity'

USE_INTERNAL_EXOG = True 
EXO_DIM_INTERNAL = 3 

# ------------------------------------------------------
# pipeline step names
DAILY_AGGREGATOR = 'daily_aggregator'
MISSING_TIME_INTS_FILLER = 'missing_time_intervals_filler'
RESHAPER_TO_THREE_D = 'reshape_to_3d'
TIME_PIVOTER = 'time_pivoter'
INDEX_SETTER = 'index_setter'
SERIES_SUBSAMPLER = 'series_subsampling'
SERIES_TRIMMER = 'series_length_trimmer'
LEFT_RIGHT_FLIPPER = 'series_left_right_flipper'
SERIES_SHUFFLER = 'series_shuffler'
MINMAX_SCALER = 'min_max_scaler'
XY_SPLITTER = 'x_y_splitter'

MOY_FEATURE_ADDER = 'add_moy_feature'
WKDAY_FEATURE_ADDER = 'add_weekday_feature'
POM_FEATURE_ADDER = 'add_pom_feature'
DUMMY_SP_DAY_FEATURE_ADDER = 'add_dummy_sp_day_feature'
DAY_NUM_FEATURE_ADDER = 'add_day_num_feature'
DAY_NUM_SCALED_ADDER = 'add_day_num_scale_feature'
DAY_NUM_MAX_SCALER = 'max_scale_day_num'
UNITY_COL_ADDER = 'add_unity_col'


NUM_REPS_PRETRAINING_DATA = 10
MAX_SCALER_UPPER_BOUND = 3.5
# ------------------------------------------------------