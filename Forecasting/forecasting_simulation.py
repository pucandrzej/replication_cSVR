"""
File contains the code for point forecasts methods (cSVR, SVR, LASSO and RF) backtest on continuous market prices from the DE intraday market.
"""

import pandas as pd
import numpy as np
import os
import sys
from numba import njit
from datetime import datetime, timedelta
import argparse
import time
from multiprocessing import Pool, RawArray
import scipy
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from numba import jit as nb_jit

# import custom functions
from remove_zerovar import remove_zerovar

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="kernel_hr_naive_mult", help="Model for forecasting."
)
parser.add_argument(
    "--daterange_start", default="2020-01-01", help="Start of eval data."
)
parser.add_argument("--daterange_end", default="2020-12-31", help="End of eval data.")
parser.add_argument("--lookback", default=365 + 62, help="Training window length.")
parser.add_argument(
    "--delivery_time",
    default=32,
    help="Index from 0 to 95 of the delivery quarter of the day.",
)
parser.add_argument(
    "--forecasting_horizons",
    default=[30, 60, 90, 120, 150, 180, 210, 300, 390, 480],
    help="Minutes from the forecast to the trade that we want to forecast price for.",
)
parser.add_argument(
    "--trade_time",
    default=32 * 15 + 8 * 60 - 60,
    help="Minute before the delivery that we want to trade in.",
)
parser.add_argument(
    "--calibration_window_len",
    default=28,
    help="Calibration window length. The logic assumes at least one day of calibration window.",
)

parser.add_argument("--processes", default=1, help="No of parallel workers to run.")

parser.add_argument(
    "--special_results_directory",
    default=None,
    help="Running on WCSS Wroclaw University of Science and Technology supercomputers requires us to save the results in dedicated path.",
)

# read the args from args parser
args = parser.parse_args()
forecasting_horizons = args.forecasting_horizons
model = args.model
start = args.daterange_start
end = args.daterange_end
lookback = int(args.lookback)
trade_time = int(args.trade_time)
delivery_time = int(args.delivery_time)
calibration_window_len = int(args.calibration_window_len)

# define the arbitrarily set hyperparameters of the models
dates = pd.date_range(start, end)  # test window dates
svr_epsilon = 0.1  # (c)SVR parameters
C = 1.0
q_kernel = 0.75
q_kernel_naive = 0.75
q_data = 0.5
q_data_naive = 0.75
lasso_max_iterations = 1000  # lasso parameters
lasso_cv_window_days = 14
random_forest_n_estimators = 256  # random forest parameters
random_forest_max_depth = 8
preprocess_option = 0  # 0 - standardize, 1 - normalize to [0,1]
remove_zero_var = True  # remove the zero variance variables from the sets S1 S2 S3
s1_variables_set_corr_threshold = 0.8
s2_variables_set_corr_threshold = 0.95
# Below are the legacy arguments to keep the results files naming convention
window_len = "expanding"
weighting_alpha = 1
variable_set = 11

# define the folder name to store the results - it will be created automatically
results_folname = "TEST_NEW_INPUT_DATA"

# prepare the directories structure to save the simulation results
if args.special_results_directory is not None:
    results_folname = os.path.join(args.special_results_directory, results_folname)
if not os.path.exists(os.path.join(results_folname)):
    os.mkdir(os.path.join(results_folname))
for forecasting_horizon in forecasting_horizons:
    results_subdirectory = f"{model}_{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}"
    if not os.path.exists(
        os.path.join(
            results_folname,
            results_subdirectory,
        )
    ):
        os.mkdir(
            os.path.join(
                results_folname,
                results_subdirectory,
            )
        )

# calibration window dates
dates_calibration = pd.date_range(
    pd.to_datetime(start) - timedelta(days=calibration_window_len),
    pd.to_datetime(start) - timedelta(days=1),
)

##################################################################
# Load the exogenous variables:
##################################################################

# DA quarterhourly prices
DA_qtrly = pd.read_csv(
    "../Data/Day-Ahead-Quarterly-Data/DA_prices_qtrly_2018_2020_preprocessed.csv",
    index_col=0,
)
DA_qtrly.index = pd.to_datetime(DA_qtrly.index)
DA_qtrly = DA_qtrly[DA_qtrly.index >= datetime(year=2018, month=10, day=31)]

# ID quarterhourly prices
ID_qtrly = pd.read_csv(
    "../Data/ID_auction_preprocessed/ID_auction_price_2018-2020_preproc.csv",
    index_col=0,
    parse_dates=True,
)
ID_qtrly = ID_qtrly[ID_qtrly.index >= datetime(year=2018, month=10, day=31)]

# load consumption
Load = pd.read_csv(
    "../Data/Load/Load_2018-2020.csv", parse_dates=["Time from"], index_col=0
)
Load.index = pd.to_datetime(Load.index)
Load = Load[Load.index >= datetime(year=2018, month=10, day=31)]

# res generation
gen = pd.read_csv("../Data/Generation/Generation_2018-2020.csv", index_col=0)
gen.index = pd.to_datetime(gen.index)
gen = gen[gen.index >= datetime(year=2018, month=10, day=31)]

# cross border trade with FR
ge_fr = pd.read_csv(
    "../Data/Crossborder/crossborder_ge_fr_2018-2020.csv",
    parse_dates=["Time from"],
    index_col=0,
)
ge_fr.index = pd.to_datetime(ge_fr.index)
ge_fr = ge_fr[ge_fr.index >= datetime(year=2018, month=10, day=31)]


@nb_jit(nopython=True)
def my_mae(X, Y):
    return np.mean(np.abs(X - Y))


def inverse_transformation(
    pred,
    naive,
    Y_raw_training_std,
    Y_raw_training_mean,
    Y_raw_training_min,
    Y_raw_training_max,
):
    if preprocess_option == 0:
        inversely_transformed_pred = (
            pred * Y_raw_training_std + Y_raw_training_mean + naive
        )
    elif preprocess_option == 1:
        inversely_transformed_pred = (
            pred * (Y_raw_training_max - Y_raw_training_min)
            + Y_raw_training_min
            + naive
        )

    return inversely_transformed_pred


# a global dictionary storing the variables passed from the initializer - we use it to pass the same data between workers
var_dict = {}


def init_worker(Model_data, Model_data_shape):
    """Used as a supplementary function to send big numpy array between workers."""
    var_dict["Model_data"] = Model_data
    var_dict["Model_data_shape"] = Model_data_shape


def run_one_day(inp):
    """Runs the main simulation for certain model, delivery and day. Saves resulting forecasts in a csv file."""
    idx = inp[0]
    date_fore = inp[1]
    forecasting_horizon = inp[2]
    model = inp[3]
    calibration_flag = inp[4]

    daily_data_window = np.swapaxes(
        np.frombuffer(var_dict["Model_data"]).reshape(var_dict["Model_data_shape"]),
        1,
        2,
    )[:idx, :, :]  # swapaxes needed after migration from table to np array

    # difference the data, assume 20min is the data avail. delay (source: dataset description given by provider)
    X = (
        daily_data_window[
            :, :-1, 20 + forecasting_horizon : -(20 + forecasting_horizon)
        ]
        - daily_data_window[:, :-1, : -2 * (20 + forecasting_horizon)]
    )  # load data without dummies
    dummies_col = daily_data_window[:, -1, -1]  # load dummies to a separate array

    Y = (
        daily_data_window[:, delivery_time, -1]
        - daily_data_window[:, delivery_time, -(20 + forecasting_horizon)]
    )  # described variable (forecasted price array)

    # extract the volume for forecasted delivery
    volume_unaggr_undiff = daily_data_window[
        :, 96 + delivery_time, 20 + forecasting_horizon : -(20 + forecasting_horizon)
    ]
    # aggregate the traded volume using a sum
    X = np.hstack((X[:, :96, :], np.expand_dims(np.sum(X[:, 96:, :], 1), 1)))

    # train the model & forecast
    results = pd.DataFrame()

    X_exog = X[
        :, -1:, :
    ]  # declare set of all the exogenous variables (starting with agg volume)

    X_close = X[
        :, np.max([delivery_time - 4, 0]) : np.min([delivery_time + 1, 96]), -60:
    ]

    # you can define your variables set here
    X = X[
        :,
        np.max([delivery_time - 8, 0]) : np.min([delivery_time + 5, 96]),
        np.append(np.arange(0, np.shape(X)[-1] - 1, 15), np.shape(X)[-1] - 1),
    ]

    # exclude the variables that have 0 variance in the forecasted day
    X, removed_zerovar_foreday_X = remove_zerovar(X)
    X_close, removed_zerovar_foreday_X_close = remove_zerovar(X_close)

    X = X.reshape(
        np.shape(X)[0], np.prod(np.shape(X)[-2:])
    )  # reshape the X to vector Days x Variables no.
    X_close = X_close.reshape(np.shape(X_close)[0], np.prod(np.shape(X_close)[-2:]))[
        7:, :
    ]

    # rewrite dummies to the linear form for the LASSO usage
    linear_dummies = np.zeros((np.shape(X)[0], 7))
    for d in range(np.shape(X)[0]):
        linear_dummies[d, int(dummies_col[d])] = 1

    # stack X vector and naive forecast (last known price without differencing)
    X = np.hstack(
        (
            X,
            np.expand_dims(
                daily_data_window[
                    -len(X) :, delivery_time, -(20 + forecasting_horizon)
                ],
                1,
            ),
        )
    )

    # add the lagged prices deltas from 1 to 7 days before for the delivery we forecast
    list_of_ar = []
    for ar_idx in range(7):
        list_of_ar.append(np.expand_dims(Y[ar_idx : -7 + ar_idx], 1))
    X = np.hstack([X[7:, :]] + list_of_ar)

    if preprocess_option == 0:
        scaler = StandardScaler()
    elif preprocess_option == 1:
        scaler = MinMaxScaler()

    # add the last known aggregated volume information
    X = np.hstack((X, X_exog[-len(X) :, :, -1]))

    delivery_hour = int(delivery_time * 0.25)  # hour in which the delivery is
    first_delivery_minute = int(
        (delivery_time * 0.25 - delivery_hour) * 4 * 15
    )  # first minute of delivery

    # add load and generation forecasts
    Load_fore = Load[
        (Load.index.hour == delivery_hour)
        & (Load.index.minute == first_delivery_minute)
    ][1 : 1 + idx]["Forecast"]
    X = np.hstack((X, np.expand_dims(Load_fore[-len(X) :], 1)))
    Gen_fore = (
        gen[
            (gen.index.hour == delivery_hour)
            & (gen.index.minute == first_delivery_minute)
        ][1 : 1 + idx]["SPV DA"]
        + gen[
            (gen.index.hour == delivery_hour)
            & (gen.index.minute == first_delivery_minute)
        ][1 : 1 + idx]["W DA"]
    )
    X = np.hstack((X, np.expand_dims(Gen_fore[-len(X) :], 1)))

    # add an actual DA price for the delivery time
    DA_price = DA_qtrly[
        (DA_qtrly.index.hour == delivery_hour)
        & (DA_qtrly.index.minute == first_delivery_minute)
    ][1 : 1 + idx]["Price"]
    X = np.hstack((X, np.expand_dims(DA_price[-len(X) :], 1)))

    # add an actual ID price for the delivery time
    DA_price = ID_qtrly[
        (ID_qtrly.index.hour == delivery_hour)
        & (ID_qtrly.index.minute == first_delivery_minute)
    ][1 : 1 + idx]["price"]
    X = np.hstack((X, np.expand_dims(DA_price[-len(X) :], 1)))

    # add the last known actual load, load forecast error, actual gen and gen forecast error and actual DE-FR exchange
    exog_avail_mins = (
        trade_time - forecasting_horizon
    )  # 61mins to account for data avail

    for exog_idx, exog_df in enumerate([Load, gen, ge_fr]):
        if exog_idx != 2:
            shift = 61
        else:
            shift = 121
        datetime_avail = (pd.to_datetime(date_fore) - timedelta(days=1)).replace(
            hour=16
        ) + timedelta(minutes=exog_avail_mins - shift)
        if (
            datetime_avail.date() < date_fore.date()
        ):  # case in which we forecast for the next day delivery
            exog_df = exog_df[exog_df.index <= pd.to_datetime(date_fore.date())]
        else:
            exog_df = exog_df[
                exog_df.index <= pd.to_datetime(date_fore.date() + timedelta(days=1))
            ]

        if exog_idx != 2:
            exog_last_info = exog_df[
                (exog_df.index.hour == datetime_avail.hour)
                & (exog_df.index.minute == datetime_avail.minute // 15 * 15)
            ]
        else:
            exog_last_info = exog_df[exog_df.index.hour == datetime_avail.hour]

        if exog_idx == 0:
            X = np.hstack((X, np.expand_dims(exog_last_info["Actual"][-len(X) :], 1)))
            exog_last_info_err = exog_last_info["Actual"] - exog_last_info["Forecast"]
            X = np.hstack((X, np.expand_dims(exog_last_info_err[-len(X) :], 1)))
        elif exog_idx == 1:
            X = np.hstack(
                (
                    X,
                    np.expand_dims(
                        exog_last_info["SPV"][-len(X) :]
                        + exog_last_info["W"][-len(X) :],
                        1,
                    ),
                )
            )
            exog_last_info_err = (
                exog_last_info["SPV"]
                + exog_last_info["W"]
                - exog_last_info["SPV DA"]
                - exog_last_info["W DA"]
            )
            X = np.hstack((X, np.expand_dims(exog_last_info_err[-len(X) :], 1)))
        elif exog_idx == 2:
            X = np.hstack(
                (
                    X,
                    np.expand_dims(
                        exog_last_info[exog_last_info.columns[0]][-len(X) :], 1
                    ),
                )
            )
    # add the sum of volume traded for forecasted delivery in the last 60 minutes
    X = np.hstack(
        (X, np.expand_dims(np.sum(volume_unaggr_undiff[-len(X) :, -60:], 1), 1))
    )
    # add the number of minutes with trades in them in the last 60 minutes
    X = np.hstack(
        (X, np.expand_dims(np.sum(volume_unaggr_undiff[-len(X) :, -60:] > 0, 1), 1))
    )
    # prepare a set of exog for kernel regression
    X_exog_fundamental = X[:, -12:]
    X_exog_fundamental = np.hstack(
        (X_exog_fundamental, dummies_col[-len(X_exog_fundamental) :, np.newaxis])
    )  # add the nonlinear dummies column

    X_exog_fundamental_plus_price = np.hstack(  # add the naive to X exog
        (
            X_exog_fundamental,
            np.expand_dims(
                daily_data_window[
                    -len(X_exog_fundamental) :,
                    delivery_time,
                    -(20 + forecasting_horizon),
                ],
                1,
            ),
        )
    )
    X_exog_fundamental_plus_price = np.hstack(  # add the differenced naive to X exog
        (
            X_exog_fundamental_plus_price,
            np.expand_dims(
                daily_data_window[
                    -len(X_exog_fundamental_plus_price) :,
                    delivery_time,
                    -(20 + forecasting_horizon),
                ]
                - daily_data_window[
                    -len(X_exog_fundamental_plus_price) :,
                    delivery_time,
                    -2 * (20 + forecasting_horizon),
                ],
                1,
            ),
        )
    )

    X_exog_fundamental_plus_price = scaler.fit_transform(X_exog_fundamental_plus_price)

    if model != "lasso":  # add the nonlinear dummies
        X = np.hstack((X, dummies_col[-len(X) :, np.newaxis]))

    X = scaler.fit_transform(X)

    # add the linearized dummies - they do not need to be standardized
    if model == "lasso":
        X = np.hstack((X, linear_dummies[-len(X) :]))

    shift_for_rejection = 0

    # ensure we do not filter exog variables using correlation filter
    if model != "lasso":
        shift_for_rejection = 21
    else:  # 6 more dummies columns are present for LASSO due to linear dummies
        shift_for_rejection = 27

    corr = np.corrcoef(X, rowvar=False)  # remove columns with correlation => threshold
    p = np.argwhere(np.triu(np.abs(corr) >= s1_variables_set_corr_threshold, 1))
    p = p[p[:, 1] <= np.shape(X)[1] - shift_for_rejection]
    X = np.delete(X, p[:, 1], axis=1)

    if np.shape(X_close)[1] > 0:  # if length X_close is nonzero: transform, else copy X
        X_close = scaler.fit_transform(X_close)
        corr = np.corrcoef(X_close, rowvar=False)
        p = np.argwhere(np.triu(np.abs(corr) >= s2_variables_set_corr_threshold, 1))
        X_close = np.delete(X_close, p[:, 1], axis=1)

    else:  # edge case when there is no data in X_close variables set - we default to X
        X_close = X.copy()

    # define constants for transformation and inverse transformation
    Y_raw_training = Y[:-1]
    Y_raw_training_std = np.std(Y_raw_training)
    Y_raw_training_mean = np.mean(Y_raw_training)
    Y_raw_training_min = np.min(Y_raw_training)
    Y_raw_training_max = np.max(Y_raw_training)
    inverse_transformation_naive = daily_data_window[
        -1, delivery_time, -(20 + forecasting_horizon)
    ]

    # transform the target
    if preprocess_option == 0:
        Y_standarized = (Y_raw_training - Y_raw_training_mean) / Y_raw_training_std
    elif preprocess_option == 1:
        Y_standarized = (Y_raw_training - Y_raw_training_min) / (
            Y_raw_training_max - Y_raw_training_min
        )
    Y_standarized = Y_standarized[-len(X) + 1 :]

    # remove 0 variance variables
    if (
        np.shape(X[:, : np.shape(X)[1] - shift_for_rejection])[1] > 0
    ):  # might happen if we remove the 0 variance elements (for example from Sun to Mon)
        selector = VarianceThreshold().fit(X[:, : np.shape(X)[1] - shift_for_rejection])
        X[:, : np.shape(X)[1] - shift_for_rejection] = selector.transform(
            X[:, : np.shape(X)[1] - shift_for_rejection]
        )

    if (
        np.shape(X_close)[1] > 0
    ):  # might happen if we remove the 0 variance elements (for example from Sun to Mon)
        try:
            # remove 0 variance variables from close set and if it is only 0 variance substitute the close set
            selector = VarianceThreshold().fit(X_close)
            X_close = selector.transform(X_close)
            if np.shape(X_close)[1] == 0:
                X_close = X.copy()
        except Exception as err:
            print(
                f"Failed to remove zero vairance variables from S2 set (X_close). Exception: {err}"
            )

    if (
        model == "lasso" and np.shape(X)[1] + lasso_cv_window_days > np.shape(X)[0]
    ):  # apply the corr filter 2nd time if LASSO and no. of variables is > than no. of samples
        no_of_variables_to_reject = (
            np.shape(X)[1] + lasso_cv_window_days - np.shape(X)[0]
        )
        corr = np.corrcoef(X, rowvar=False)
        highly_correlated_variables = np.unravel_index(
            np.argsort(np.triu(np.abs(corr), 1), axis=None),
            np.triu(np.abs(corr), 1).shape,
        )[1]
        highly_correlated_variables = highly_correlated_variables[
            highly_correlated_variables <= np.shape(X)[1] - shift_for_rejection
        ][::-1]
        _, idx = np.unique(highly_correlated_variables, return_index=True)
        X = np.delete(
            X,
            highly_correlated_variables[np.sort(idx)][:no_of_variables_to_reject],
            axis=1,
        )

    # forecast with LASSO model
    if args.model == "lasso":
        # prepare a time series appropriate CV setting
        train_idxs = []
        test_idxs = []
        for cv_test_idx in range(lasso_cv_window_days):  # test on the last two weeks
            train_idxs.append(
                list(range(0, len(Y_standarized) - lasso_cv_window_days + cv_test_idx))
            )
            test_idxs.append(len(Y_standarized) - lasso_cv_window_days + cv_test_idx)
        # PREDICT S1
        reg = LassoLarsCV(
            cv=zip(train_idxs, test_idxs),
            max_iter=lasso_max_iterations,
            fit_intercept=False,  # fit_intercept is false as we already fit dummies which sum to 1 in every index of training window
        ).fit(X[:-1, :], Y_standarized)
        pred = reg.predict(X[np.newaxis, -1, :])
        results["insample MAE"] = [my_mae(reg.predict(X[:-1, :]), Y_standarized)]
        results["prediction"] = inverse_transformation(
            pred,
            inverse_transformation_naive,
            Y_raw_training_std,
            Y_raw_training_mean,
            Y_raw_training_min,
            Y_raw_training_max,
        )
        # PREDICT S2
        reg = LassoLarsCV(
            cv=zip(train_idxs, test_idxs),
            max_iter=lasso_max_iterations,
            fit_intercept=False,
        ).fit(X_close[:-1, :], Y_standarized)
        pred = reg.predict(X_close[np.newaxis, -1, :])
        results["insample MAE close"] = [
            my_mae(reg.predict(X_close[:-1, :]), Y_standarized)
        ]
        results["prediction_close"] = inverse_transformation(
            pred,
            inverse_transformation_naive,
            Y_raw_training_std,
            Y_raw_training_mean,
            Y_raw_training_min,
            Y_raw_training_max,
        )
        # PREDICT S3
        reg = LassoLarsCV(
            cv=zip(train_idxs, test_idxs),
            max_iter=lasso_max_iterations,
            fit_intercept=False,
        ).fit(X_exog_fundamental_plus_price[:-1, :], Y_standarized)
        pred = reg.predict(X_exog_fundamental_plus_price[np.newaxis, -1, :])
        results["insample MAE exog"] = [
            my_mae(reg.predict(X_exog_fundamental_plus_price[:-1, :]), Y_standarized)
        ]
        results["prediction_exog"] = inverse_transformation(
            pred,
            inverse_transformation_naive,
            Y_raw_training_std,
            Y_raw_training_mean,
            Y_raw_training_min,
            Y_raw_training_max,
        )

    # Forecast with Random Forest model
    elif model == "random_forest":
        # PREDICT S1
        regr = RandomForestRegressor(
            n_estimators=random_forest_n_estimators, max_depth=random_forest_max_depth
        )
        regr.fit(X[:-1, :], Y_standarized)
        pred = regr.predict(X[np.newaxis, -1, :])
        results["insample MAE"] = [my_mae(regr.predict(X[:-1, :]), Y_standarized)]
        results["prediction"] = inverse_transformation(
            pred,
            inverse_transformation_naive,
            Y_raw_training_std,
            Y_raw_training_mean,
            Y_raw_training_min,
            Y_raw_training_max,
        )
        # PREDICT S2
        regr = RandomForestRegressor(
            n_estimators=random_forest_n_estimators, max_depth=random_forest_max_depth
        )
        regr.fit(X_close[:-1, :], Y_standarized)
        pred = regr.predict(X_close[np.newaxis, -1, :])
        results["insample MAE close"] = [
            my_mae(regr.predict(X_close[:-1, :]), Y_standarized)
        ]
        results["prediction_close"] = inverse_transformation(
            pred,
            inverse_transformation_naive,
            Y_raw_training_std,
            Y_raw_training_mean,
            Y_raw_training_min,
            Y_raw_training_max,
        )
        # PREDICT S3
        regr = RandomForestRegressor(
            n_estimators=random_forest_n_estimators, max_depth=random_forest_max_depth
        )
        regr.fit(X_exog_fundamental_plus_price[:-1, :], Y_standarized)
        pred = regr.predict(X_exog_fundamental_plus_price[np.newaxis, -1, :])
        results["insample MAE exog"] = [
            my_mae(regr.predict(X_exog_fundamental_plus_price[:-1, :]), Y_standarized)
        ]
        results["prediction_exog"] = inverse_transformation(
            pred,
            inverse_transformation_naive,
            Y_raw_training_std,
            Y_raw_training_mean,
            Y_raw_training_min,
            Y_raw_training_max,
        )

    # Forecast with SVR/cSVR models
    elif model == "kernel_hr_naive_mult":
        training_window = X[
            :-1, :
        ]  # define the training windows to prepare training kernel matrices
        training_window_close = X_close[:-1, :]
        training_window_fundamental_plus_price = X_exog_fundamental_plus_price[:-1, :]
        naive_vec = daily_data_window[7:, delivery_time, -(20 + forecasting_horizon)]

        if preprocess_option == 0:
            naive_vec_standardized = (naive_vec - np.mean(naive_vec)) / np.std(
                naive_vec
            )
        elif preprocess_option == 1:
            naive_vec_standardized = (naive_vec - np.min(naive_vec)) / (
                np.max(naive_vec) - np.min(naive_vec)
            )

        @njit
        def calc_interm_kernel(interm_data, norm: int = 2):
            """Calculate the intermediate kernel, i.e. the matrix of distances between two vectors of variables."""
            plain_kernel = np.zeros((len(interm_data), len(interm_data)))
            for i in range(len(interm_data)):
                for j in range(len(interm_data)):
                    if i < j:
                        if norm == 2:
                            plain_kernel[i, j] = np.sqrt(
                                np.sum((interm_data[i] - interm_data[j]) ** 2)
                            )  # L2 distance
                        elif norm == 1:
                            plain_kernel[i, j] = np.sum(
                                np.abs(interm_data[i] - interm_data[j])
                            )  # L1 distance
            plain_kernel = plain_kernel + plain_kernel.T
            return plain_kernel

        # prepare the intermediate kernels, i.e. norms matrices
        plain_kernel_exog_plus_prices_train = calc_interm_kernel(
            training_window_fundamental_plus_price
        )
        plain_kernel_exog_plus_prices_test = np.sqrt(
            np.sum(
                (
                    training_window_fundamental_plus_price
                    - X_exog_fundamental_plus_price[np.newaxis, -1, :]
                )
                ** 2,
                axis=1,
            )
        )
        plain_kernel_X_1_train = calc_interm_kernel(training_window)
        plain_kernel_close_train = calc_interm_kernel(training_window_close)
        plain_kernel_X_1_test = np.sqrt(
            np.sum((training_window - X[np.newaxis, -1, :]) ** 2, axis=1)
        )
        plain_kernel_close_test = np.sqrt(
            np.sum((training_window_close - X_close[np.newaxis, -1, :]) ** 2, axis=1)
        )
        plain_kernel_X_2_test = (
            np.abs(naive_vec_standardized[:-1] - naive_vec_standardized[np.newaxis, -1])
            ** 2
        )  # such formulation can be used because in 1D the Euclidean distance is just an abs value raised to power 2
        plain_kernel_X_2_train = (
            np.abs(
                np.tile(
                    naive_vec_standardized[:-1], [len(naive_vec_standardized[:-1]), 1]
                )
                - np.swapaxes(
                    np.tile(
                        naive_vec_standardized[:-1],
                        [len(naive_vec_standardized[:-1]), 1],
                    ),
                    0,
                    1,
                )
            )
            ** 2
        )

        kernels_train = [
            plain_kernel_X_1_train,
            plain_kernel_close_train,
            plain_kernel_exog_plus_prices_train,
            plain_kernel_X_2_train,
        ]

        kernels_test = [
            plain_kernel_X_1_test,
            plain_kernel_close_test,
            plain_kernel_exog_plus_prices_test,
            plain_kernel_X_2_test,
        ]

        # L1 S_1 kernels for laplace SVR input to compare with L2
        plain_kernel_X_1_train_L1 = calc_interm_kernel(training_window, norm=1)

        plain_kernel_X_1_test_L1 = np.sum(
            np.abs(training_window - X[np.newaxis, -1, :]), axis=1
        )

        def fast_calculate_kernel_matrix(stage, kernel_option=0):
            """Calculates the full kernel matrix by applying exponents and weights to the distance matrix."""
            sigma = (
                np.quantile(kernels_train[-1], q_data_naive)
                / scipy.stats.norm.ppf(q_kernel_naive, loc=0, scale=1)
            )  # parameters are ALWAYS based on the train window so thet we use the same kernel

            if stage == "train":
                calc_kernels = kernels_train
                calc_kernel_L1 = plain_kernel_X_1_train_L1
            elif stage == "test":
                calc_kernels = kernels_test
                calc_kernel_L1 = plain_kernel_X_1_test_L1

            if kernel_option == 1:
                width = np.log(2 - 2 * q_kernel) / np.quantile(kernels_train[0], q_data)
                return np.exp(
                    width * calc_kernels[0] - 1 / (2 * sigma**2) * calc_kernels[-1]
                )
            elif kernel_option == 2:
                width = np.log(2 - 2 * q_kernel) / np.quantile(kernels_train[1], q_data)
                return np.exp(
                    width * calc_kernels[1] - 1 / (2 * sigma**2) * calc_kernels[-1]
                )
            elif (
                kernel_option == 7
            ):  # exog with added info on naive and differenced naive
                width = np.log(2 - 2 * q_kernel) / np.quantile(kernels_train[2], q_data)
                return np.exp(
                    width * calc_kernels[2] - 1 / (2 * sigma**2) * calc_kernels[-1]
                )
            elif kernel_option == "plain_laplace_L2_1":
                width = np.log(2 - 2 * q_kernel) / np.quantile(kernels_train[0], q_data)
                return np.exp(width * calc_kernels[0])
            elif kernel_option == "plain_laplace_L2_2":
                width = np.log(2 - 2 * q_kernel) / np.quantile(kernels_train[1], q_data)
                return np.exp(width * calc_kernels[1])
            elif kernel_option == "plain_laplace_L2_3":
                width = np.log(2 - 2 * q_kernel) / np.quantile(kernels_train[2], q_data)
                return np.exp(width * calc_kernels[2])
            elif kernel_option == "plain_laplace":
                width = np.log(2 - 2 * q_kernel) / np.quantile(
                    plain_kernel_X_1_train_L1, q_data
                )
                return np.exp(width * calc_kernel_L1)

        # train and forecast (c)SVR for every variables set and kernel configuration
        for kernel_choice in [
            1,  # S1
            2,  # S2
            7,  # S3 (legacy name from testing procedures performed at early stage of development)
            "plain_laplace_L2_1",
            "plain_laplace_L2_2",
            "plain_laplace_L2_3",
        ]:
            training_matrix = fast_calculate_kernel_matrix(
                stage="train", kernel_option=kernel_choice
            )
            estimator = SVR(kernel="precomputed", epsilon=svr_epsilon, C=C)
            estimator.fit(training_matrix, Y_standarized)
            results[f"insample MAE_{kernel_choice}"] = [
                my_mae(estimator.predict(training_matrix), Y_standarized)
            ]
            test_matrix = fast_calculate_kernel_matrix(
                stage="test", kernel_option=kernel_choice
            )
            pred = estimator.predict(test_matrix.reshape(1, -1))
            results[f"prediction_{kernel_choice}"] = inverse_transformation(
                pred,
                inverse_transformation_naive,
                Y_raw_training_std,
                Y_raw_training_mean,
                Y_raw_training_min,
                Y_raw_training_max,
            )

    results["actual"] = [daily_data_window[-1, delivery_time, -1]]
    results["naive"] = [inverse_transformation_naive]

    try:
        results.to_csv(
            f"{results_folname}/{model}_{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}/{calibration_flag}_{str((pd.to_datetime(date_fore) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=trade_time)).replace(':', ';')}_{forecasting_horizon}_{variable_set}_weights_{weighting_alpha}_window_{window_len}.csv"
        )
    except Exception as err:  # raise to avoid saving an empty file
        os.remove(
            f"{results_folname}/{model}_{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}/{calibration_flag}_{str((pd.to_datetime(date_fore) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=trade_time)).replace(':', ';')}_{forecasting_horizon}_{variable_set}_weights_{weighting_alpha}_window_{window_len}.csv"
        )
        raise KeyboardInterrupt(
            f"Interrupted on saving: last file removed to avoid empty files. Exception: {err}."
        )


if __name__ == "__main__":
    # load the continuous intraday market data
    con = sqlite3.connect("data_ID.db")
    sql_str = f"SELECT * FROM with_dummies WHERE Index_daily <= {trade_time};"  # load only the data required for simu, so up to trade time
    daily_data = pd.read_sql(sql_str, con)[[str(i) for i in range(193)]].to_numpy()
    daily_data = np.reshape(
        daily_data, (np.shape(daily_data)[0] // trade_time, trade_time, 193)
    )

    # scan the directory for existing files - it will automatically skip them if these are in currently specified results folder
    print("Checking for simulation results that are already saved...")
    run_simu = False
    required_forecasting_horizons = []
    required_dates = {}
    required_indices = {}
    required_dates_calibration = {}
    required_indices_calibration = {}

    for forecasting_horizon in forecasting_horizons:
        required_dates[forecasting_horizon] = {}
        required_indices[forecasting_horizon] = {}
        required_dates_calibration[forecasting_horizon] = {}
        required_indices_calibration[forecasting_horizon] = {}
        if np.shape(daily_data)[1] <= 2 * (20 + forecasting_horizon):
            print(
                (
                    f"Differencing could not be performed with series of length {np.shape(daily_data)[1]}min and horizon {forecasting_horizon}min (+20min of data avail delay.)"
                )
            )
            continue

        required_dates[forecasting_horizon] = []
        required_indices[forecasting_horizon] = []
        required_dates_calibration[forecasting_horizon] = []
        required_indices_calibration[forecasting_horizon] = []
        for date_idx, date in enumerate(dates):
            if not os.path.isfile(
                f"{results_folname}/{model}_{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}/test_{str((pd.to_datetime(date) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=trade_time)).replace(':', ';')}_{forecasting_horizon}_{variable_set}_weights_{weighting_alpha}_window_{window_len}.csv"
            ):
                run_simu = True

                if forecasting_horizon not in required_forecasting_horizons:
                    required_forecasting_horizons.append(forecasting_horizon)
                required_dates[forecasting_horizon].append(date)
                required_indices[forecasting_horizon].append(date_idx)
        for date_idx, date in enumerate(dates_calibration):
            if not os.path.isfile(
                f"{results_folname}/{model}_{start}_{end}_{lookback}_{delivery_time}_{[forecasting_horizon]}_{trade_time}_{remove_zero_var}/calibration_{str((pd.to_datetime(date) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=trade_time)).replace(':', ';')}_{forecasting_horizon}_{variable_set}_weights_{weighting_alpha}_window_{window_len}.csv"
            ):
                required_dates_calibration[forecasting_horizon].append(date)
                required_indices_calibration[forecasting_horizon].append(date_idx)
        if run_simu:
            print(
                f"Config fore. horizon: {forecasting_horizon}, variable set: {variable_set} needs a recalculation."
            )
    if not run_simu:
        print("All required config was already simulated. Exitting.")
        sys.exit(0)

    raw_arr = RawArray(
        "d", np.shape(daily_data)[0] * np.shape(daily_data)[1] * np.shape(daily_data)[2]
    )

    # wrap X as an numpy array so we can easily manipulates its data in multiprocessing scheme
    daily_data_np = np.frombuffer(raw_arr).reshape(np.shape(daily_data))
    # copy data to a shared array.
    np.copyto(daily_data_np, daily_data)
    data_shape = np.shape(daily_data)

    # free memory of the unneccessary data
    daily_data = None
    del daily_data

    simu_start = time.time()
    for forecasting_horizon in required_forecasting_horizons:
        # configure the calibration window forecasts (needed for weights estimation in weighting averaging scheme)
        inputlist_calibration = [
            [
                lookback - calibration_window_len + idx,
                date,
                forecasting_horizon,
                model,
                "calibration",
            ]
            for idx, date in zip(
                required_indices_calibration[forecasting_horizon],
                required_dates_calibration[forecasting_horizon],
            )
        ]
        # run the calibration window simulation
        with Pool(
            processes=int(args.processes),
            initializer=init_worker,
            initargs=(raw_arr, data_shape),
        ) as p:
            _ = p.map(run_one_day, inputlist_calibration)
        # configure the test window simulation
        inputlist = [
            [
                lookback + idx,
                date,
                forecasting_horizon,
                model,
                "test",
            ]
            for idx, date in zip(
                required_indices[forecasting_horizon],
                required_dates[forecasting_horizon],
            )
        ]
        # run the test window simulation
        with Pool(
            processes=int(args.processes),
            initializer=init_worker,
            initargs=(raw_arr, data_shape),
        ) as p:
            _ = p.map(run_one_day, inputlist)

    simu_end = time.time()
    print(simu_end - simu_start, "Total time of simulation:")
