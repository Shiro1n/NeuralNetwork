import numpy as np


def mean_squared_error(predicted, expected):
    return np.mean((predicted - expected) ** 2)


def mean_squared_error_gradient(predicted, expected):
    N = predicted.shape[0]
    return 2 * (predicted - expected) / N


def log_loss(predicted, expected):
    predicted = np.clip(predicted, 1e-15, 1 - 1e-15)  # Prevent log(0)
    return -np.mean(expected * np.log(predicted) + (1 - expected) * np.log(1 - predicted))


def log_loss_gradient(predicted, expected):
    predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
    return (predicted - expected) / (predicted * (1 - predicted))


def categorical_cross_entropy_loss(predicted, expected):
    predicted = np.clip(predicted, 1e-15, 1 - 1e-15)  # Prevent log(0)
    return -np.sum(expected * np.log(predicted)) / predicted.shape[0]


def categorical_cross_entropy_loss_gradient(predicted, expected):
    predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
    return -(expected / predicted) / predicted.shape[0]