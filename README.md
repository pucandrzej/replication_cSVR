# Replication package for "Corrected Support Vector Regression for intraday point forecasting of prices in the continuous power market"

## Authors
Andrzej Puć, Joanna Janczura

Wrocław University of Science and Technology, Faculty of Pure and Applied Mathematics, Hugo Steinhaus Center, Wyb. Wyspiańskiego 27, Wrocław, 50-370, Poland

### Contact information
andrzej.puc@pwr.edu.pl

## Date of replication package creation
2025.10.27

## Overview & contents
The code in this replication material allows for recalculating the forecasting simulation which served as an illustration of forecasting methodology proposed in the paper "Corrected Support Vector Regression for intraday point forecasting of prices in the continuous power market". 
When the simulation is recalculated, each figure presented in the paper can be generated using `paper_figures_reproduction.ipynb` file. The notebook saves generated figures in the Paper_Figures directory.

Alternatively, one can generate figures from the paper using precomputed intermediary files by running the notebook right after downloading the repository.
**Please note that raw data used in the forecasting study is not fully publicly available. Thus, Figures 1, 2, A.9 and A.10 cannot be generated based solely on the contents of this repository.**

## Software requirements
These analyses were run on Python 3.11.
**Following packages are required to regenerate the paper figures**: 

`numpy==1.23.5`
`pandas==2.0.3`
`matplotlib==3.7.2`

while a full list of packages needed for recalculating the simulation can be found in the requirements.txt file.

A full LaTeX installation is required. Requirements for text rendering with LaTeX in Matplotlib can be found here: [link](https://matplotlib.org/stable/users/explain/text/usetex.html).

## Data availability and provenance
The raw data is stored in the `Data` directory. In this repository it contains the exogenous variables used in the forecasting study: crossborder physical flows, day-ahead quarter-hourly German market electricity prices, SPV and wind generation actual values and forecasts and load actual values and forecasts.
All of the aforementioned data was sourced from ENTSOe. Necessary links and time of accessing the data are provided in `META.txt` files in each subdirectory.

Two non-public directories are not attached in `Data`: `ID_auction_preprocessed/` and `Transactions/`. Data stored in these directories is a part of a package DE Trades on the continuous market - Histo (up to Y-1):
https://webshop.eex-group.com/data-type/de-trades-continuous-market-histo-y-1. The data has been purchased from the EXPEX Spot under University License, under which the Contracting Party is entitled to a limited Internal Usage in unchanged format according to Section 3 of the General Conditions, specifically for educational and academic research purposes and publication of results of analysis and research. The Agreement with the EPEX Spot do not allow to transfer the data to third Parties. It can be accessed through EPEX Spot sFTP server. The yearly cost of this access is equal to 480EUR.

## Hardware requirements and expected runtime
The simulation relies on heavy usage of parallel computing.
It was performed using the resources of Wrocław Centre for Networking and Supercomputing (WCSS).
Specifically, CPU: 2 x Intel Xeon Platinum 8268 (24 cores, 2,9 GHz), RAM: 192 GB 2933 MHz ECC DDR4.
Runtime on such config, using 48 parallel workers, is around 2.5 days for (c)SVR models simulation and roughly 14 days for the limited (only for lead time of 60min) LASSO and RF simulations respectively.

## Running the simulation
**If you only want to regenerate figures from the paper, it is enough to run the `paper_figures_reproduction.ipynb` notebook.**
If your goal is to run the complete simulation, please follow the steps below.

### Preprocess the data
Store the downloaded EPEX Spot continuous market transactions in yearly directories in `Data/Transactions/` folder/ In this study, these are `2018`, `2019` and `2020` directories containing daily `.csv` files with transations corresponding to this delivery date.

Run the `preprocess_transactions.py` to preprocess the data in line with preprocessing approach described in the paper.

If you also want to change the horizon and use different ranges of exogenous variables downloaded from ENTSOe, you can use the `` script for dst handling.
This step can be skipped if you keep the original range of simulation.

### Run the simulation
Run the `simulation_runner.py` script to schedule all of the simulations.
You can adjust the concurrent workers by applying changes in `forecasting_config.py` script.

### Add the weighted averages to the raw forecasts
Having the forecasts, you can augment them with their weighted average using `intel_avg_generator.py`.

### Calculate MAE/QAPE aggregations and run the Diebold-Mariano test
Finally, the forecasts can be analyzed using accuracy measures and Diebold-Mariano test.

For that, run the `mae_aggregator.py` script.


After completing these steps, you can run the `paper_figures_reproduction.ipynb` on your own results.
