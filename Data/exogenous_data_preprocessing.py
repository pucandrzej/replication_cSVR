"""
Handling the dst and missing data in the exogenous variables.
The unexpected missing data was detected only in day-ahead load forecast from ENTSOe.
These missing day-ahead load forecasts cover a period of 25 days in 2018 and are handled by imputation of
actual + noise, where the noise is sampled for each quarter-hour separately from 30d lookback period.
Other variables only need the dst handling.
"""

import os

try:
    os.chdir("Data")  # for debuger run
except:
    pass
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

required_start = datetime(
    2018, 10, 31
)  # we cannot go back due to the lack of Load forecasts in few days there
required_end = datetime(2021, 1, 1)


def fill_march_dst(df, col):
    # get the dst gaps
    dst_gaps = df[(df.index >= required_start) & df[col].isna()].index
    print(dst_gaps)

    # Fill missing values using average of Hour 2 and Hour 3 (before and after gap)
    for ts in dst_gaps:
        before = ts - pd.Timedelta(hours=1)
        after = ts + pd.Timedelta(hours=1)
        if before in df.index and after in df.index:
            df.loc[ts] = (df.loc[before] + df.loc[after]) / 2

    return df


def fill_forecasts(
    load_df, actual="Actual", forecast="Forecast", lookback_days=30, seed=None
):
    """Fills the missing day-ahead forecasts that are missing in the ENTSOe data using sampled historical errors wrt actual"""
    rng = np.random.default_rng(seed)
    df = load_df.copy()

    # weekday type (bus vs weekend)
    wd = df.index.weekday < 5
    weekday_type = np.where(wd, "bus", "weekend")

    # quarter-hour group key
    qh = df.index.hour * 4 + df.index.minute // 15

    # only original (non-imputed) rows become donor pool
    donor = df[~df[forecast].isna()].copy()
    donor["err"] = donor[actual] - donor[forecast]
    donor["wk"] = np.where(donor.index.weekday < 5, "bus", "weekend")
    donor["qh"] = donor.index.hour * 4 + donor.index.minute // 15

    # through manual investigation we know that there are no missing actuals, thus we can use them in imputation logic
    for t in df.index[df[forecast].isna() & df[actual].notna()]:
        t0 = t - pd.Timedelta(days=lookback_days)
        pool = donor.loc[
            (donor.index >= t0)
            & (donor.index < t)
            & (donor["wk"] == weekday_type[df.index.get_loc(t)])
            & (donor["qh"] == qh[df.index.get_loc(t)])
            & (~donor["err"].isna())
        ]

        # it can happen that we do not have enough history - then we allow for future errors usage (4 such cases detected - negligible for forecasting study results)
        if pool.empty:
            pool = donor.loc[
                (donor.index >= t0)
                & (donor.index < t + pd.Timedelta(days=14))
                & (donor["wk"] == weekday_type[df.index.get_loc(t)])
                & (donor["qh"] == qh[df.index.get_loc(t)])
                & (~donor["err"].isna())
            ]
        if pool.empty:
            raise ValueError("Empty donors pool!")

        sampled_err = rng.choice(pool["err"].values)

        df.at[t, forecast] = (
            df.at[t, actual] - sampled_err
        )  # put the sampled value in place of NaN forecast

    return df


###################################################################################
print("Preprocessing the day-ahead auction prices...")
dayahead = []
for year in range(2018, 2021):
    dayahead.append(
        pd.read_csv(
            os.path.join("Day-Ahead-Quarterly-Data", f"Day-ahead Prices_{year}.csv"),
            na_values=["n/e"],
        )
    )

dayahead = pd.concat(dayahead, ignore_index=True)

new_datetimes = []
for dat in dayahead["MTU (CET/CEST)"]:
    new_datetimes.append(dat.split(" - ")[0])

dayahead_df = pd.DataFrame()
dayahead_df["Time from"] = pd.to_datetime(
    new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
)
dayahead_df["Price"] = dayahead["Day-ahead Price [EUR/MWh]"]
dayahead_df.index = pd.to_datetime(dayahead_df["Time from"])
dayahead_df.drop(columns="Time from", inplace=True)

# remove October dst change impact by deduplication and fill March dst NaNs with avg of adjacent periods
dayahead_df = fill_march_dst(dayahead_df[~dayahead_df.index.duplicated()], col="Price")

dayahead_df.to_csv(
    "Day-Ahead-Quarterly-Data/DA_prices_qtrly_2018_2020_preprocessed.csv"
)

print(
    f"Len matching the required len: {len(dayahead_df[dayahead_df.index >= required_start]) == len(pd.date_range(required_start, required_end, inclusive='left', freq='15min'))}"
)
print(
    f"Any NaNs remaining? {dayahead_df[dayahead_df.index >= required_start].isnull().values.any()}"
)

###################################################################################
print("Processing the intraday auction data...")
total_df = pd.concat(
    [
        pd.read_csv(
            "ID_auction_preprocessed/intraday_auction_spot_prices_15-call-DE_2018.csv",
            skiprows=1,
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        ),
        pd.read_csv(
            "ID_auction_preprocessed/intraday_auction_spot_prices_15-call-DE_2019.csv",
            skiprows=1,
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        ),
        pd.read_csv(
            "ID_auction_preprocessed/intraday_auction_spot_prices_15-call-DE_2020.csv",
            skiprows=1,
            index_col=0,
            parse_dates=True,
            dayfirst=True,
        ),
    ]
).iloc[:, :-8]
# drop the second (duplicated) DST delivery
total_df = total_df[[c for c in total_df.columns if "Hour 3B" not in c]]

# prepare df of desired format
new_df = []
for idx in total_df.index:
    new_df.append(
        pd.DataFrame(
            columns=["price"],
            index=pd.date_range(idx, idx + timedelta(minutes=95 * 15), freq="15min"),
            data=total_df.loc[idx].to_numpy(),
        )
    )

# fill the missing hour in march DST by avg of adjacent hours
df = fill_march_dst(pd.concat(new_df).sort_index(), col="price")
df.to_csv("ID_auction_preprocessed/ID_auction_price_2018-2020_preproc.csv")

print(
    f"Len matching the required len: {len(df[df.index >= required_start]) == len(pd.date_range(required_start, required_end, inclusive='left', freq='15min'))}"
)
print(f"Any NaNs remaining? {df[df.index >= required_start].isnull().values.any()}")

###################################################################################
print("Processing the hourly exchange data...")
ge_fr = []
for year in range(2018, 2021):
    ge_fr.append(
        pd.read_csv(f"Crossborder/crossborder_ge_fr_{year}.csv", na_values=["n/e"])
    )

ge_fr = pd.concat(ge_fr, ignore_index=True)

new_datetimes = []
for dat in ge_fr["Time (CET/CEST)"]:
    new_datetimes.append(dat.split(" - ")[0])

ge_fr_df = pd.DataFrame()
ge_fr_df["Time from"] = pd.to_datetime(
    new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
)
ge_fr_df["DE > FR"] = (
    ge_fr["BZN|DE-LU > BZN|FR [MW]"] - ge_fr["BZN|FR > BZN|DE-LU [MW]"]
)
ge_fr_df.index = pd.to_datetime(ge_fr_df["Time from"])
ge_fr_df.drop(columns="Time from", inplace=True)

# remove dst change impact
ge_fr_df = fill_march_dst(ge_fr_df[~ge_fr_df.index.duplicated()], col="DE > FR")

ge_fr_df.to_csv("Crossborder/crossborder_ge_fr_2018-2020.csv")

print(
    f"Len matching the required len: {len(ge_fr_df[ge_fr_df.index >= required_start]) == len(pd.date_range(required_start, required_end, inclusive='left', freq='1H'))}"
)
print(
    f"Any NaNs remaining? {ge_fr_df[ge_fr_df.index >= required_start].isnull().values.any()}"
)

###################################################################################
print("Processing the load data...")
load = []
for year in range(2018, 2021):
    load.append(
        pd.read_csv(
            f"Load/Total Load - Day Ahead _ Actual_{year}.csv", na_values=["n/e"]
        )
    )

load = pd.concat(load, ignore_index=True)

new_datetimes = []
for dat in load["Time (CET/CEST)"]:
    new_datetimes.append(dat.split(" - ")[0])

load_df = pd.DataFrame()
load_df["Time from"] = pd.to_datetime(
    new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
)
load_df["Actual"] = load["Actual Total Load [MW] - BZN|DE-LU"]
load_df["Forecast"] = load["Day-ahead Total Load Forecast [MW] - BZN|DE-LU"]
load_df.index = pd.to_datetime(load_df["Time from"])
load_df.drop(columns="Time from", inplace=True)

# remove October dst change impact
load_df = fill_march_dst(load_df[~load_df.index.duplicated()], col="Actual")

# special handling of missing day-ahead forecasts through imputation of synthetic forecasts generated based on the adjacent forecasts errors
filled_load = fill_forecasts(load_df, seed=0)
filled_load.to_csv("Load/Load_2018-2020.csv")

print(
    f"Len matching the required len: {len(filled_load[filled_load.index >= required_start]) == len(pd.date_range(required_start, required_end, inclusive='left', freq='15min'))}"
)
print(
    f"Any NaNs remaining? {filled_load[filled_load.index >= required_start].isnull().values.any()}"
)

###################################################################################
print("Processing the generation data...")
gen = []
for year in range(2018, 2021):
    gen.append(pd.read_csv(f"Generation/generation_{year}.csv", na_values=["n/e"]))

gen = pd.concat(gen, ignore_index=True)[
    [
        "MTU",
        "Solar  - Actual Aggregated [MW]",
        "Wind Offshore  - Actual Aggregated [MW]",
        "Wind Onshore  - Actual Aggregated [MW]",
    ]
]

new_datetimes = []
for dat in gen["MTU"]:
    new_datetimes.append(dat.split(" - ")[0])

gen_df = pd.DataFrame()
gen_df["Time from"] = pd.to_datetime(
    new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
)
gen_df["SPV"] = gen["Solar  - Actual Aggregated [MW]"]
gen_df["W"] = (
    gen["Wind Offshore  - Actual Aggregated [MW]"]
    + gen["Wind Onshore  - Actual Aggregated [MW]"]
)
gen_df.index = pd.to_datetime(gen_df["Time from"])
gen_df.drop(columns="Time from", inplace=True)

gen_fore = []
for year in range(2018, 2021):
    gen_fore.append(
        pd.read_csv(f"Generation/generation_fore_{year}.csv", na_values=["n/e"])
    )

gen_fore = pd.concat(gen_fore, ignore_index=True)[
    [
        "MTU (CET/CEST)",
        "Generation - Solar  [MW] Day Ahead/ BZN|DE-LU",
        "Generation - Wind Offshore  [MW] Day Ahead/ BZN|DE-LU",
        "Generation - Wind Onshore  [MW] Day Ahead/ BZN|DE-LU",
    ]
]

new_datetimes = []
for dat in gen_fore["MTU (CET/CEST)"]:
    new_datetimes.append(dat.split(" - ")[0])

gen_fore_df = pd.DataFrame()
gen_fore_df["Time from"] = pd.to_datetime(
    new_datetimes, format="%d.%m.%Y %H:%M", dayfirst=False, yearfirst=False
)
gen_fore_df["SPV DA"] = gen_fore["Generation - Solar  [MW] Day Ahead/ BZN|DE-LU"]
gen_fore_df["W DA"] = (
    gen_fore["Generation - Wind Offshore  [MW] Day Ahead/ BZN|DE-LU"]
    + gen_fore["Generation - Wind Onshore  [MW] Day Ahead/ BZN|DE-LU"]
)
gen_fore_df.index = pd.to_datetime(gen_fore_df["Time from"])
gen_fore_df.drop(columns="Time from", inplace=True)

gen_df = pd.concat([gen_df, gen_fore_df], axis=1)

gen_df = fill_march_dst(gen_df[~gen_df.index.duplicated()], col="SPV")

print(
    f"Len matching the required len: {len(gen_df[gen_df.index >= required_start]) == len(pd.date_range(required_start, required_end, inclusive='left', freq='15min'))}"
)
print(
    f"Any NaNs remaining? {gen_df[gen_df.index >= required_start].isnull().values.any()}"
)

gen_df.to_csv("Generation/Generation_2018-2020.csv")
