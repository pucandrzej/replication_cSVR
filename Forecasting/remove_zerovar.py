import numpy as np


def remove_zerovar(X):
    """Remove variables with 0 variance throughout the training window."""
    removed_zerovar_foreday = []
    if len(np.where(~X[-1, :, :].any(axis=1))[0]) > 0:
        removed_zerovar_foreday.append(np.where(~X[-1, :, :].any(axis=1))[0])
        return np.delete(
            X, np.where(~X[-1, :, :].any(axis=1))[0], 1
        ), removed_zerovar_foreday
    else:
        return X, removed_zerovar_foreday
