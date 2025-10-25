"""
Script to run the simulation for the required configuration of distances before trading and forecast and different deliveries
"""

import time
import subprocess
import sys

from forecasting_config import forecasting_config

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--start_delivery", default=0, help="Start of the simulated deliveries"
)
parser.add_argument(
    "--end_delivery", default=96, help="End + 1 of the simulated deliveries"
)
parser.add_argument(
    "--models",
    default=["kernel_hr_naive_mult", "lasso", "random_forest"],
    help="Models to simulate.",
)
parser.add_argument(
    "--calibration_window_len",
    default=28,
    help="For every date consider a historical results from a calibration window.",
)
parser.add_argument(
    "--special_results_directory",
    default=None,
    help="Running on WCSS Wroclaw University of Science and Technology supercomputers requires us to save the results in dedicated path.",
)
args = parser.parse_args()

# additional parameters
processes = forecasting_config["default_parallel_workers"]

for model in args.models:
    start = args.start_delivery
    joblist = []
    sys.stderr = open(
        f"TOTAL_SIMU_ERR_{start}_{args.end_delivery}_{model}_{args.calibration_window_len}.txt",
        "w",
    )
    sys.stdout = open(
        f"TOTAL_SIMU_LOG_{start}_{args.end_delivery}_{model}_{args.calibration_window_len}.txt",
        "w",
    )
    for shift_trade in forecasting_config[
        "lead_times"
    ]:  # delivery time - shift_trade is the trade time
        if shift_trade != 60 and model != "kernel_hr_naive_mult":
            continue
        for delivery_time in range(int(args.start_delivery), int(args.end_delivery)):
            trade_time = delivery_time * 15 + 8 * 60 - shift_trade
            joblist.append(
                [
                    "python",
                    "forecasting_simulation.py",
                    "--model",
                    model,
                    "--trade_time",
                    str(trade_time),
                    "--delivery_time",
                    str(delivery_time),
                    "--calibration_window_len",
                    str(args.calibration_window_len),
                    "--processes",
                    str(processes),
                    "--special_results_directory",
                    args.special_results_directory,
                ]
            )

    invoked = 0
    stack = []
    ts = time.time()
    concurrent = 1
    while invoked < len(joblist):
        while len(stack) == concurrent:
            for no, p in enumerate(stack):
                if p.poll() is not None:
                    stack.pop(no)
                    break
            time.sleep(1)
        line = joblist[invoked]
        print(f"running job {invoked + 1} of {len(joblist)}: {joblist[invoked]}")
        stack.append(subprocess.Popen(line, stderr=sys.stderr, stdout=sys.stdout))
        stack[-1].wait()  # wait for the process to finish
        invoked += 1
