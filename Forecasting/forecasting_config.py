"""General configuration of the forecasting simulation."""

import os
import numpy as np

forecasting_config = {
    "forecasting_horizons": [480, 390, 300, 210, 180, 150, 120, 90, 60, 30],
    "lead_times": [30, 60, 90, 120, 150, 180],
    "deliveries": np.arange(96),
    "default_parallel_workers": 32,
    "forecasting_results_dir": os.path.join("RESULTS", "cSVR_SVR_LASSO_RF_FORECASTS"),
    "mae_aggregation_results_dir": os.path.join("RESULTS", "cSVR_LASSO_RF_MAE"),
}
