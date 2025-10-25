# calculates the MAE of kernel models in parallel
import pandas as pd
import numpy as np
import os
import pickle
from multiprocessing import Pool
from scipy import stats
from datetime import datetime

from forecasting_config import forecasting_config

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--errors_choice",
    default="standard",
    help="The choice of errors for simulation: 'standard' takes into account all of the errors, while 'extreme' requires passing the 'extreme_surprise_quantile' threshold for errors (market surprises).",
)
parser.add_argument(
    "--extreme_surprise_quantile",
    default=0.25,
    help="Float, extreme surprise quantile. Default is 0.25. It is then taken from below and above.",
)
args = parser.parse_args()

results_dir = forecasting_config[
    "mae_aggregation_results_dir"
]  # directory to save the results of MAE/QAPE analysis


def mae_qape(Y, X, naive, measure_type="avg"):
    """MAE if measure_type is avg,"""
    if measure_type == "avg":
        return np.mean(np.abs(X - Y))
    elif measure_type.startswith("quantile_"):
        q = measure_type.split("_")[1]
        assert q.replace(".", "", 1).isdigit(), (
            f"Quantile needs to be a float, got {q}"
        )  # check if quantile is valid
        return np.quantile(np.abs(X - Y), float(q))
    else:
        raise ValueError(f"No measure defined for measure_type = {measure_type}.")


def dm_pval(actual, naive, col_results):
    """p-value of Diebold-Mariano test"""
    d = np.abs(naive - actual) - np.abs(col_results - actual)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=0)
    DM_stat = mean_d / np.sqrt((1 / len(d)) * var_d)
    return 1 - stats.norm.cdf(DM_stat)


def prepare_mae_per_delivery(inp):
    """Loading the results and calculating MAE/QAPE for certain delivery and for each lead time and forecasting horizon.
    The resulting MAE is saved in pickle files.
    """
    delivery, horizons, forecasts_dir = inp
    extreme_quantile = float(args.extreme_surprise_quantile)
    for measure_type in ["avg", "quantile_0.5", "quantile_0.25", "quantile_0.75"]:
        for trade_vs_delivery_delta in forecasting_config["lead_times"]:
            mae_results = {}
            for horizon in horizons:
                mae_results[horizon] = {}
                trade_time = delivery * 15 + 8 * 60 - trade_vs_delivery_delta
                mae_results[horizon][trade_time] = {}

                fore_dir = f"kernel_hr_naive_mult_2020-01-01_2020-12-31_427_{delivery}_[{horizon}]_{trade_time}_True"
                random_forest_dir = f"random_forest_2020-01-01_2020-12-31_427_{delivery}_[{horizon}]_{trade_time}_True"
                lasso_dir = f"lasso_2020-01-01_2020-12-31_427_{delivery}_[{horizon}]_{trade_time}_True"

                if os.path.exists(os.path.join(forecasts_dir, fore_dir)):
                    forecasts = [
                        f
                        for f in os.listdir(os.path.join(forecasts_dir, fore_dir))
                        if ".csv" in f and "test_" in f
                    ]

                    if len(forecasts):
                        df_sample = pd.read_csv(
                            os.path.join(forecasts_dir, fore_dir, forecasts[0])
                        )

                        actual = []
                        naive = []
                        all_dates = []
                        for fore in forecasts:
                            all_dates.append(
                                datetime.strptime(
                                    fore.split("_")[1].split(" ")[0], "%Y-%m-%d"
                                )
                            )
                            df = pd.read_csv(
                                os.path.join(forecasts_dir, fore_dir, fore)
                            )
                            try:
                                actual.append(df.loc[0, "actual"])
                                naive.append(df.loc[0, "naive"])
                            except Exception as err:
                                raise ValueError(
                                    f"Failed to load the results file {os.path.join(forecasts_dir, fore_dir, fore)}. Exception: {err}"
                                )
                        actual = np.array(actual)
                        naive = np.array(naive)
                        all_dates = np.array(all_dates)

                        if (
                            args.errors_choice == "standard"
                        ):  # consider all of the prices
                            extreme_indices = range(len(actual))
                        else:  # get only the extreme quantiles of price
                            relative_surprise = np.abs(actual - naive)

                            if extreme_quantile > 0.5:
                                extreme_indices = (
                                    relative_surprise
                                    > np.quantile(  # will set False in place of comparison with NaN
                                        relative_surprise, extreme_quantile
                                    )
                                )
                            else:
                                extreme_indices = (
                                    relative_surprise
                                    < np.quantile(  # will set False in place of comparison with NaN
                                        relative_surprise, extreme_quantile
                                    )
                                )

                        if (
                            trade_time == delivery * 15 + 8 * 60 - 60
                        ):  # only for this lead time we have LASSO and RF results
                            for col in [
                                "prediction",
                                "prediction_close",
                                "prediction_exog",
                                "prediction_32",
                            ]:
                                # Random Forest results
                                col_results = []
                                for fore in forecasts:
                                    df_random_forest = pd.read_csv(
                                        os.path.join(
                                            forecasts_dir,
                                            random_forest_dir,
                                            fore,
                                        )
                                    )
                                    col_results.append(df_random_forest.loc[0, col])
                                col_results = np.array(col_results)
                                mae_results[horizon][trade_time][
                                    col + "random_forest"
                                ] = mae_qape(
                                    actual[extreme_indices],
                                    col_results[extreme_indices],
                                    naive,
                                    measure_type,
                                )
                                # add DM test p-value
                                p_value = dm_pval(
                                    actual[extreme_indices],
                                    naive[extreme_indices],
                                    col_results[extreme_indices],
                                )
                                mae_results[horizon][trade_time][
                                    col + "_DM_wrt_naive_pval" + "random_forest"
                                ] = p_value

                                # LASSO results
                                col_results = []
                                for fore in forecasts:
                                    df_lasso = pd.read_csv(
                                        os.path.join(
                                            forecasts_dir,
                                            lasso_dir,
                                            fore,
                                        )
                                    )
                                    col_results.append(df_lasso.loc[0, col])
                                col_results = np.array(col_results)
                                mae_results[horizon][trade_time][col + "lasso"] = (
                                    mae_qape(
                                        actual[extreme_indices],
                                        col_results[extreme_indices],
                                        naive,
                                        measure_type,
                                    )
                                )
                                # add DM test p-value
                                p_value = dm_pval(
                                    actual[extreme_indices],
                                    naive[extreme_indices],
                                    col_results[extreme_indices],
                                )
                                mae_results[horizon][trade_time][
                                    col + "_DM_wrt_naive_pval" + "lasso"
                                ] = p_value

                        if len(actual) == 366:
                            fore_cols = {}
                            for col in df_sample.columns:
                                try:
                                    if "prediction" in col or col == "naive":
                                        col_results = []
                                        for fore in forecasts:
                                            df = pd.read_csv(
                                                os.path.join(
                                                    forecasts_dir, fore_dir, fore
                                                )
                                            )
                                            col_results.append(df.loc[0, col])
                                        col_results = np.array(col_results)
                                        mae_results[horizon][trade_time][col] = (
                                            mae_qape(
                                                actual[extreme_indices],
                                                col_results[extreme_indices],
                                                naive,
                                                measure_type,
                                            )
                                        )
                                        # for models other than naive add the DM test p-value
                                        if col != "naive":
                                            p_value = dm_pval(
                                                actual[extreme_indices],
                                                naive[extreme_indices],
                                                col_results[extreme_indices],
                                            )
                                            mae_results[horizon][trade_time][
                                                col + "_DM_wrt_naive_pval"
                                            ] = p_value
                                            fore_cols[col] = col_results[
                                                extreme_indices
                                            ]
                                except Exception as err:
                                    print(f"Skipped column {col}. Exception: {err}")

                # remove empty config
                if mae_results[horizon] == {}:
                    del mae_results[horizon]
                if mae_results[horizon][trade_time] == {}:
                    del mae_results[horizon][trade_time]

            # check if results have been collected and save them
            if mae_results != {}:
                if args.errors_choice == "extreme":
                    pickle.dump(
                        mae_results,
                        open(
                            os.path.join(
                                results_dir,
                                f"{args.extreme_surprise_quantile}_relative_surprise_{measure_type}_mae_results_{trade_vs_delivery_delta}_{delivery}.pickle",
                            ),
                            "wb",
                        ),
                    )
                else:
                    pickle.dump(
                        mae_results,
                        open(
                            os.path.join(
                                results_dir,
                                f"{measure_type}_mae_results_{trade_vs_delivery_delta}_{delivery}.pickle",
                            ),
                            "wb",
                        ),
                    )


if __name__ == "__main__":
    forecasts_dir = forecasting_config["forecasting_results_dir"]
    results_dirs = os.listdir(forecasts_dir)

    deliveries = forecasting_config["deliveries"]
    horizons = forecasting_config["forecasting_horizons"]

    with Pool(processes=forecasting_config["default_parallel_workers"]) as p:
        inputlist = [(i, horizons, forecasts_dir) for i in deliveries]
        _ = p.map(prepare_mae_per_delivery, inputlist)
