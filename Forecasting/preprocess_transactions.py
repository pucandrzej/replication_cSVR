import glob
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    os.chdir("Forecasting")  # for debuger run
except:
    pass

import sqlite3
import warnings
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo  # Python 3.9+

warnings.filterwarnings("ignore")


def is_date_of_DSTtransition(dt: datetime, zone: str) -> bool:
    """
    check if the date part of a datetime object falls on the date
    of a DST transition.
    """
    _d = datetime.combine(dt, time.min).replace(tzinfo=ZoneInfo(zone))
    return _d.dst() != (_d + timedelta(1)).dst()


def initial_preprocessing():
    """Clean the transactions to get the consistent datetime index and format from every year (handling reporting changes) and day (handling dst)."""
    if not os.path.exists("../Data/Transactions/price_analysis_table.csv"):
        # load the complete dataset
        if not os.path.exists("../Data/Transactions/concatenated_table.csv"):
            df = pd.concat(
                [
                    pd.read_csv(
                        f,
                        skiprows=2,
                        names=[
                            "Date",
                            "Area Buy",
                            "Market Area Sell",
                            "Hour from",
                            "Hour to",
                            "Volume (MW)",
                            "Price (EUR)",
                            "Time Stamp",
                            "Trade ID",
                        ],
                    )
                    for f in glob.glob("../Data/Transactions/*/*.csv")
                ]
            )
            df.to_csv(
                "../Data/Transactions/concatenated_table.csv",
                usecols=[
                    "Hour from",
                    "Hour to",
                    "Time Stamp",
                    "Price (EUR)",
                    "Volume (MW)",
                    "Date",
                ],
            )
        else:
            df = pd.read_csv(
                "../Data/Transactions/concatenated_table.csv",
                usecols=[
                    "Hour from",
                    "Hour to",
                    "Time Stamp",
                    "Price (EUR)",
                    "Volume (MW)",
                    "Date",
                ],
            )

        # cut only the quarter-hourly trades from the dataset & the columns that we need
        df_copy = df[
            (df["Hour from"].str.contains("qh"))
            & (~df["Hour from"].str.contains("B"))
            & (
                df["Hour from"] == df["Hour to"]
            )  # there is no such case in our dataset, but otherwise this would be sensible condition
        ].reset_index()

        # create the datetime from and datetime offer time columns
        mod_trans_from = []
        offer_time = []
        time_change = []
        for i, val in tqdm(enumerate(df_copy["Hour from"])):
            if "qh" in val and not ("A" in val or "B" in val):
                minutes = str((int(val.split("qh")[1]) - 1) * 15)
                hours = val.split("qh")[0]
                hours = str(int(hours) - 1)
                mod_trans_from.append(
                    datetime.strptime(
                        df_copy["Date"][i]
                        + " "
                        + hours.zfill(2)
                        + ":"
                        + minutes
                        + ":00",
                        "%d/%m/%Y %H:%M:%S",  # round the data to full minutes
                    )
                )
                time_change.append(0)
            if (
                "A" in val
            ):  # A is 2:00, so we need to avg its preprocessed trajectories later in the code
                minutes = str((int(val.split("Aqh")[1]) - 1) * 15)
                hours = str(int(val.split("Aqh")[0]) - 1)
                time_change.append(0)
                if datetime.strptime(
                    df_copy["Date"][i] + " " + hours.zfill(2) + ":" + minutes + ":00",
                    "%d/%m/%Y %H:%M:%S",  # round the data to full minutes
                ) < pd.to_datetime(df_copy.loc[i, "Time Stamp"]):
                    raise
                mod_trans_from.append(
                    datetime.strptime(
                        df_copy["Date"][i]
                        + " "
                        + hours.zfill(2)
                        + ":"
                        + minutes
                        + ":00",
                        "%d/%m/%Y %H:%M:%S",  # round the data to full minutes
                    )
                )

            offer_time.append(
                datetime.strptime(
                    df_copy["Time Stamp"][i][:-2] + "00", "%d/%m/%Y %H:%M:%S"
                )
            )
        df_copy["Datetime from"] = mod_trans_from
        df_copy["Datetime offer time"] = offer_time
        df_copy["Time change"] = time_change
        df_copy.to_csv("../Data/Transactions/price_analysis_table.csv")
    print(
        "Preliminary preprocessed already performed. Performing additional preprocessing..."
    )
    # read the preprocessed dataset
    df_copy = pd.read_csv("../Data/Transactions/price_analysis_table.csv")

    # transform columns to datetime from string
    df_copy["Datetime offer time"] = pd.to_datetime(df_copy["Datetime offer time"])
    df_copy["Datetime from"] = pd.to_datetime(df_copy["Datetime from"])

    # shift trades from 14:00 to 15:00 in 25.10.2020 by 1h
    df_copy.loc[
        (
            df_copy["Datetime offer time"]
            >= datetime(day=24, month=10, year=2020, hour=14)
        )
        & (
            df_copy["Datetime offer time"]
            < datetime(day=24, month=10, year=2020, hour=15)
        )
        & (df_copy["Datetime offer time"].dt.date != df_copy["Datetime from"].dt.date),
        "Datetime offer time",
    ] = df_copy.loc[
        (
            df_copy["Datetime offer time"]
            >= datetime(day=24, month=10, year=2020, hour=14)
        )
        & (
            df_copy["Datetime offer time"]
            < datetime(day=24, month=10, year=2020, hour=15)
        )
        & (df_copy["Datetime offer time"].dt.date != df_copy["Datetime from"].dt.date),
        "Datetime offer time",
    ] + timedelta(hours=1)

    # # shift winter by 1h
    df_copy.loc[
        (df_copy["Datetime from"].dt.year == 2020)
        & (
            (df_copy["Datetime from"] < datetime(day=29, month=3, year=2020, hour=3))
            | (
                df_copy["Datetime from"]
                >= datetime(day=25, month=10, year=2020, hour=3)
            )
        ),
        "Datetime offer time",
    ] = df_copy.loc[
        (df_copy["Datetime from"].dt.year == 2020)
        & (
            (df_copy["Datetime from"] < datetime(day=29, month=3, year=2020, hour=3))
            | (
                df_copy["Datetime from"]
                >= datetime(day=25, month=10, year=2020, hour=3)
            )
        ),
        "Datetime offer time",
    ] + timedelta(hours=1)

    # # shift summer by 2h
    df_copy.loc[
        (df_copy["Datetime from"] >= datetime(day=29, month=3, year=2020, hour=3))
        & (df_copy["Datetime from"] < datetime(day=25, month=10, year=2020, hour=3)),
        "Datetime offer time",
    ] = df_copy.loc[
        (df_copy["Datetime from"] >= datetime(day=29, month=3, year=2020, hour=3))
        | (df_copy["Datetime from"] < datetime(day=25, month=10, year=2020, hour=3)),
        "Datetime offer time",
    ] + timedelta(hours=2)

    # add the weekdays
    dates = pd.date_range(
        start=np.min(df_copy["Datetime offer time"]),
        end=np.max(df_copy["Datetime offer time"]),
        freq="H",
    )
    weekdays = []
    for d in dates:
        weekdays.append(d.weekday())
    weekdays = np.array(weekdays)
    week_days_col = []
    for i in tqdm(df_copy["Datetime offer time"]):
        week_days_col.append(i.weekday())
    df_copy["Week day"] = week_days_col

    # define TTD column as a difference between delivery and offer times
    df_copy["Time to delivery"] = (
        pd.to_datetime(df_copy["Datetime from"])
        - pd.to_datetime(df_copy["Datetime offer time"])
    ).dt.total_seconds() / 60

    # drop the columns unnecessary to analysis
    df_copy = df_copy.drop("Date", axis=1)
    df_copy = df_copy.drop("Time Stamp", axis=1)
    df_copy = df_copy.drop("Unnamed: 0", axis=1)
    df_copy = df_copy.drop("index", axis=1)
    df_copy = df_copy.drop("Hour from", axis=1)
    df_copy = df_copy.drop("Hour to", axis=1)

    # save the resulting table of unevenly spaced trades with volume, prices and day of the week
    df_copy.to_csv("../Data/preprocessed_dataset.csv", date_format="%s")


def preprocess_data(start, end, ID_qtrly, add_dummies):
    """Prepare the 5min averages of transactions extended by intraday auction price from the left and last known transaction price from the right."""
    demanded_len = 32 * 60  # daily data len (all minutes)
    print("Cached data unavailable, preparing & saving the data.")
    try:
        df = pd.read_csv("../Data/preprocessed_dataset.csv")
    except Exception as err:
        print(
            f"Failed to read the preprocessed_dataset.csv. Exception: {err}.\nPreparing the initially preprocessed dataset..."
        )
        initial_preprocessing()
        print("Done.")
        df = pd.read_csv("../Data/preprocessed_dataset.csv")
    df["Datetime from"] = pd.to_datetime(df["Datetime from"])
    df["Datetime offer time"] = pd.to_datetime(df["Datetime offer time"])
    df = df[(df["Datetime from"] >= start) & ((df["Datetime from"]) < end)]

    print("Done reading the initially preprocessed dataset.")

    print("Preprocessing the data...")
    index = np.arange(1, 1921)
    index_daily = np.arange(1, 1921)
    for d, date in enumerate(np.unique(df["Datetime from"].dt.date)):
        preprocessed_data = pd.DataFrame(index=index)  # daily df to sqlite update
        index = index + d * 1920  # update the index

        df_day = df[
            df["Datetime from"].dt.date == date
        ]  # frame with all trades for day d
        unique_datetime_from = np.unique(df_day["Datetime from"])
        if len(unique_datetime_from) != 96:
            print(
                f"{date} WARNING: trajectories for {96 - len(unique_datetime_from)} deliveries will be averaged using steps back and forward corresponding to no. of missing deliveries."
            )
        shift = 0  # shift the index in case a hole is detected in unique deliveries
        stable_shift = 0
        first_deliveries_unavail = False
        missing_windows = {}
        last_delivery = pd.to_datetime(np.sort(unique_datetime_from)[0]).replace(
            minute=0
        )
        for delivery_idx, delivery in enumerate(np.sort(unique_datetime_from)):
            if (
                pd.to_datetime(np.sort(unique_datetime_from)[0]).replace(minute=0)
                != last_delivery
                and (
                    pd.to_datetime(delivery) - pd.to_datetime(last_delivery)
                ).total_seconds()
                / 60
                != 15
            ):  # check whether we are not missing any deliveries
                shift = (
                    int(
                        (
                            (
                                pd.to_datetime(delivery) - pd.to_datetime(last_delivery)
                            ).total_seconds()
                            / 60
                        )
                        // 15
                    )
                    - 1
                )
                missing_windows[delivery_idx] = (
                    shift  # save the no. of missing deliveries
                )
            elif (
                delivery_idx == 0
                and pd.to_datetime(np.sort(unique_datetime_from)[0]).replace(minute=0)
                == last_delivery
                and (
                    pd.to_datetime(delivery) - pd.to_datetime(last_delivery)
                ).total_seconds()
                / 60
                != 0
            ):
                shift = (
                    int(
                        (
                            (
                                pd.to_datetime(delivery) - pd.to_datetime(last_delivery)
                            ).total_seconds()
                            / 60
                        )
                        // 15
                    )
                    - 1
                )
                missing_windows[delivery_idx] = shift
                first_deliveries_unavail = True

            last_delivery = delivery
            delivery_idx = delivery_idx + shift + stable_shift  # perform the shifting
            stable_shift += shift  # all consecutive indices are shifted by stable shift
            shift = 0  # reset the shift
            if delivery_idx == 60 and d == 5:
                fig, axs = plt.subplots(nrows=2, figsize=(20, 10))
                df_day[df_day["Datetime from"] == delivery]
                ax = axs[0]
                ax.plot(
                    df_day[df_day["Datetime from"] == delivery]["Datetime offer time"],
                    df_day[df_day["Datetime from"] == delivery]["Price (EUR)"],
                    marker=".",
                    label="raw price",
                )

                ax = axs[1]
                ax.plot(
                    df_day[df_day["Datetime from"] == delivery]["Datetime offer time"],
                    df_day[df_day["Datetime from"] == delivery]["Volume (MW)"],
                    marker=".",
                    label="raw volume",
                )

            df_day_delivery = df_day[df_day["Datetime from"] == delivery]
            current_data_avg = df_day_delivery.groupby(
                "Time to delivery", as_index=False
            )
            # PRICE PREPROCESSING
            price = []
            time_to_delivery = []
            # first known indicator of quarterhourly price is day-ahead auction trade
            for group in current_data_avg:
                price.append(
                    np.sum(
                        group[1]["Price (EUR)"].to_numpy()
                        * group[1]["Volume (MW)"].to_numpy()
                    )
                    / np.sum(group[1]["Volume (MW)"])
                )
                time_to_delivery.append(group[1]["Time to delivery"].to_numpy()[0])
            time_to_delivery = time_to_delivery[::-1]
            price = price[::-1]
            trading_start = (
                (
                    pd.to_datetime(delivery)
                    - (pd.to_datetime(delivery) - timedelta(days=1))
                    .replace(hour=16)
                    .replace(minute=0)
                ).total_seconds()
                / 60
            )  # trading starts at 16:00 each day - we compute this date and time as minutes to delivery
            if trading_start > np.max(time_to_delivery):
                price = [
                    float(ID_qtrly[ID_qtrly.index == delivery]["price"].to_numpy()[0])
                ] + price
                time_to_delivery = [trading_start] + time_to_delivery
            end = 0
            ttd = time_to_delivery + [end]
            add_nos = -np.diff(ttd)
            prices = [
                ele for i, ele in enumerate(price) for j in range(int(add_nos[i]))
            ]  # filling the missing minutes
            if len(prices) < demanded_len:
                prices = np.hstack(
                    (
                        prices,
                        np.ones(demanded_len - len(prices)) * np.mean(prices[-180:]),
                    )
                )  # approximating ID3 - maybe putting ID3 here would be better: sth to check to
            elif len(prices) > demanded_len:
                prices = prices[:demanded_len]

            if delivery_idx == 60 and d == 5:
                ax = axs[0]
                x_dates = pd.date_range(
                    (pd.to_datetime(delivery) - timedelta(days=1)).replace(hour=16),
                    (pd.to_datetime(delivery) - timedelta(days=1)).replace(hour=16)
                    + timedelta(minutes=demanded_len - 1),
                    freq="min",
                )
                ax.plot(x_dates, prices, label="preprocessed price")
                ax.set_xlabel("datetime")
                ax.set_title("Price")
                ax.legend()
                ax.set_ylabel("price [EUR]")
            preprocessed_data[delivery_idx] = prices[:demanded_len]
            # VOLUME PREPROCESSING
            volume = []
            time_to_delivery = []
            for group in current_data_avg:
                if (
                    np.sum(df_day_delivery["Time change"]) > 0
                ):  # if current hour is 2:00 and date change occurred we average the 2:00 and 3A deliveries
                    volume.append(np.sum(group[1]["Volume (MW)"]) / 2)
                else:
                    volume.append(np.sum(group[1]["Volume (MW)"]))
                time_to_delivery.append(group[1]["Time to delivery"].to_numpy()[0])
            time_to_delivery = time_to_delivery[::-1]
            volume = volume[::-1]
            trading_start = (
                pd.to_datetime(delivery)
                - (pd.to_datetime(delivery) - timedelta(days=1))
                .replace(hour=16)
                .replace(minute=0)
            ).total_seconds() / 60
            if trading_start > np.max(time_to_delivery):
                volume = [0] + volume
                time_to_delivery = [trading_start] + time_to_delivery
            end = 0
            ttd = time_to_delivery + [end]
            add_nos = -np.diff(ttd)
            volumes = [
                0 if j > 0 else ele
                for i, ele in enumerate(volume)
                for j in range(int(add_nos[i]))
            ]  # filling the missing minutes
            if len(volumes) < demanded_len:
                volumes = np.hstack(
                    (volumes, np.zeros(demanded_len - len(volumes)))
                )  # fill with total traded volume
            elif len(volumes) > demanded_len:
                volumes = volumes[:demanded_len]

            if delivery_idx == 60 and d == 5:
                ax = axs[1]
                ax.plot(x_dates, volumes, label="preprocessed volume")
                ax.set_xlabel("datetime")
                ax.set_ylabel("price [EUR]")
                ax.set_title("Volume")
                ax.legend()
                plt.savefig(f"sample_preprocessing_{delivery_idx}_{d}.pdf")
                plt.close(fig)
            preprocessed_data[delivery_idx + 96] = volumes[:demanded_len]

        preprocessed_data = preprocessed_data.reindex(
            sorted(preprocessed_data.columns), axis=1
        )
        for corr_idx in missing_windows.keys():
            if first_deliveries_unavail and corr_idx == 0:
                for col_n in range(corr_idx, corr_idx + missing_windows[corr_idx]):
                    preprocessed_data[col_n] = preprocessed_data[
                        col_n + missing_windows[corr_idx]
                    ]
                    preprocessed_data[col_n + 96] = preprocessed_data[
                        col_n + missing_windows[corr_idx] + 96
                    ]
            else:
                for col_n in range(corr_idx, corr_idx + missing_windows[corr_idx]):
                    preprocessed_data[col_n] = (
                        preprocessed_data[col_n - missing_windows[corr_idx]]
                        + preprocessed_data[col_n + missing_windows[corr_idx]]
                    ) / 2
                    preprocessed_data[col_n + 96] = (
                        preprocessed_data[col_n - missing_windows[corr_idx] + 96]
                        + preprocessed_data[col_n + missing_windows[corr_idx] + 96]
                    ) / 2

        # ADD DUMMIES
        preprocessed_data[96 + 96] = np.ones(demanded_len) * date.weekday()

        preprocessed_data["Time"] = pd.date_range(
            (pd.to_datetime(date) - timedelta(days=1))
            .replace(hour=16)
            .replace(minute=0),
            pd.to_datetime(date) + timedelta(days=1),
            freq="1min",
            inclusive="left",
        )
        preprocessed_data["Index_daily"] = index_daily
        preprocessed_data["Day"] = np.ones(np.shape(index_daily)) * d
        preprocessed_data.to_sql("with_dummies", con, if_exists="append")

        print(f"Done {d} of {len(np.unique(df['Datetime from'].dt.date))}")


if __name__ == "__main__":
    if not os.path.exists("data_ID.db"):
        con = sqlite3.connect("data_ID.db")
    else:
        os.remove("data_ID.db")
        con = sqlite3.connect("data_ID.db")

    ID_qtrly = pd.read_csv(
        "../Data/ID_auction_preprocessed/ID_auction_price_2018-2020_preproc.csv",
        index_col=0,
        parse_dates=True,
    )

    preprocess_data(
        datetime(2018, 11, 1, 0, 0, 0),
        datetime(2021, 1, 2, 0, 0, 0),
        ID_qtrly,
        True,
    )
    con.close()
