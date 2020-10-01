# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss_MSE(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    return np.sum(np.square(y-tx@w))/(y.shape[0]*2)

def compute_loss_MAE(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    return np.sum(abs(y-tx@w))/(y.shape[0])