
import numpy as np
from sklearn.metrics import mean_squared_error as mse_func, r2_score, mean_absolute_percentage_error
# from sklearn.utils import check_arrays



def get_mse(Y, Yhat):
    return mse_func(Y, Yhat)


def get_smape(Y, Yhat):
    return 100./len(Y) * np.sum(2 * np.abs(Yhat - Y) / (np.abs(Y) + np.abs(Yhat)))


def get_mape(Y, Yhat): 
    # Y, Yhat = check_arrays(Y, Yhat)
    # return np.mean(np.abs((Y - Yhat) / Y)) * 100
    return mean_absolute_percentage_error(Y, Yhat)


def get_wape(Y, Yhat): 
    abs_diff = np.abs(Y - Yhat)
    return 100 * np.sum(abs_diff) / np.sum(Y)



def get_r_squared(Y, Yhat): 
    return r2_score(Y, Yhat)


if __name__ == '__main__':

    Y = np.array([2,3,4,5,6,7,8,9])
    Yhat = np.array([1,3,5,4,6,7,10,7])

    mse = get_mse(Y, Yhat)
    print(f'mse: {mse}')

    smape = get_smape(Y, Yhat)
    print(f'smape: {smape}')

    mape = get_mape(Y, Yhat)
    print(f'mape: {mape}')

