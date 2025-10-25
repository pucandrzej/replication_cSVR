"""
postprocessing file to get the MAE weighted forecasts
scripts that generates the intel avg (avg based on the MAE on calibration window)
calculate the MAE of each model in the calibration window
"""

import os

import shutil
import pandas as pd
from datetime import timedelta
import numpy as np
from multiprocessing import Pool

from forecasting_config import forecasting_config

results_dir = forecasting_config["forecasting_results_dir"]
cols_sets_to_average = [["prediction_1", "prediction_2", "prediction_7", "naive"]]
models = ["kernel_hr_naive_mult", "lasso", "random_forest"]

calibration_window_lens = [
    7,
    14,
    21,
    28,
]  # test of several calibration windows for the intel. avg
dates = pd.date_range("2020-01-01", "2020-12-31")  # test window dates


def my_mae(X, Y):
    return np.mean(np.abs(X - Y))


def load_delivery_results(inp):
    delivery, horizons = inp
    print(f"Processing: {delivery}")
    for model in models:
        for trade_vs_delivery_delta in forecasting_config["lead_times"]:
            for forecasting_horizon in horizons:
                col_idx = 20  # LEGACY col index
                for cols_to_average in cols_sets_to_average:
                    if (
                        model != "kernel_hr_naive_mult"
                    ):  # for LASSO and RF avg only the existing columns corresponding to
                        cols_to_average = [
                            "prediction",
                            "prediction_close",
                            "prediction_exog",
                            "naive",
                        ]

                    for calibration_window_len in calibration_window_lens:
                        trade_time = delivery * 15 + 8 * 60 - trade_vs_delivery_delta
                        calib_weights_mul = {}
                        try:
                            for test_date in dates:
                                forecast_frames = []
                                for calib_date in pd.date_range(
                                    test_date - timedelta(calibration_window_len),
                                    test_date - timedelta(1),
                                ):
                                    if calib_date < pd.to_datetime(
                                        "2020-01-01"
                                    ):  # covered by a calibration run
                                        calibration_flag = "calibration"
                                    else:
                                        calibration_flag = "test"

                                    forecast_frames.append(
                                        pd.read_csv(
                                            os.path.join(
                                                results_dir,
                                                f"{model}_2020-01-01_2020-12-31_427_{delivery}_[{forecasting_horizon}]_{trade_time}_True",
                                                f"{calibration_flag}_{str((pd.to_datetime(calib_date) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=int(trade_time))).replace(':', ';')}_{forecasting_horizon}_11_weights_1.0_window_expanding.csv",
                                            ),
                                        )
                                    )

                                # calculate the MAE errors from the calibration window
                                errors = []
                                for col_to_avg in cols_to_average:
                                    forecast = []
                                    actual = []
                                    for i_df, df in enumerate(forecast_frames):
                                        actual.append(df["actual"][0])
                                        forecast.append(df[col_to_avg][0])
                                    errors.append(
                                        my_mae(np.array(forecast), np.array(actual))
                                    )

                                # derive the multiplicative weights based on the forecast errors
                                weights = []
                                for i in range(len(errors)):
                                    weights.append(
                                        (1 / errors[i]) / np.sum(1 / np.array(errors))
                                    )

                                calib_weights_mul[test_date] = weights

                            # for every forecast in test window create the average forecasts based on the weights
                            for _, date in enumerate(dates):
                                base_path = os.path.join(
                                    results_dir,
                                    f"{model}_2020-01-01_2020-12-31_427_{delivery}_[{forecasting_horizon}]_{trade_time}_True",
                                    f"test_{str((pd.to_datetime(date) - timedelta(days=1)).replace(hour=16) + timedelta(minutes=int(trade_time))).replace(':', ';')}_{forecasting_horizon}_11_weights_1.0_window_expanding.csv",
                                )

                                forecast = pd.read_csv(
                                    base_path,
                                    index_col=0,
                                )

                                avg_result = forecast.copy()

                                weights = calib_weights_mul[date]
                                avg_result[
                                    f"prediction_{12 + col_idx}"
                                ] = [  # LEGACY index shift by 12
                                    np.sum(
                                        avg_result[cols_to_average].to_numpy()[0]
                                        * np.array(weights)
                                    )
                                ]

                                avg_result.to_csv(
                                    base_path.replace(".csv", "_TMP.csv"),
                                )

                                # remove the old base file
                                if os.path.exists(base_path):
                                    os.remove(base_path)
                                # move the tmp to base file
                                shutil.move(
                                    base_path.replace(".csv", "_TMP.csv"), base_path
                                )

                            col_idx += 1
                        except Exception as err:
                            print(
                                f"Failed to process for calibration window {calibration_window_len} due to {err}."
                            )


if __name__ == "__main__":
    deliveries = forecasting_config["deliveries"]

    horizons = forecasting_config["forecasting_horizons"]

    inputlist = [(i, horizons) for i in deliveries]

    with Pool(processes=forecasting_config["default_parallel_workers"]) as p:
        _ = p.map(load_delivery_results, inputlist)
