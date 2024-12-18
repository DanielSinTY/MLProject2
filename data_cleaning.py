import numpy as np
import pandas as pd

def get_first_n_periods(x, k):
    """
    Returns the first n periods of a worm's data.
    
    Parameters:
        x : list of numpy.Array
            N x C x D matrix containing N worms, C time frames, and D features values

        k : int
            The number of periods to return
    
    Returns:
        x: numpy.Array
            N x k x 900 x D numpy array representing the xy coordinates of the first k periods of each worm
    """

    x_trimmed = [matrix[:900*k,[2,3]].reshape((-1,900,2)) for matrix in x]
    x_trimmed = np.stack(x_trimmed, axis=0)
    return x_trimmed

def remove_na(threshold_proportion, x, y):
    """
    Removes worms with more than a certain proportion of missing data.

    Parameters:
        threshold_proportion : float
            The proportion of missing data above which a worm is removed

        x : numpy.Array
            N x k x 900 x D numpy array representing the xy coordinates of the first k periods of each worm

        y : numpy.Array
            N x 1 array of labels 
    
    Returns:
        x : numpy.Array
            N_1 x k x 900 x D numpy array representing the xy coordinates of the first k periods of each worm

        y : numpy.Array
            N_1 x 1 array of labels 
    """

    total_size = x.shape[1]*x.shape[2]
    mask = np.any(np.isnan(x).sum(axis=(1,2))/total_size < threshold_proportion,axis=1)

    return x[mask], y[mask],mask

def remove_outliers(threshold, x, y):
    """
    Removes worms with unreasonable lifespan values.

    Parameters:
        threshold : float
            The threshold on the value in y below which a worm is removed

        x : numpy.Array
            N x k x 900 x D numpy array representing the xy coordinates of the first k periods of each worm

        y : numpy.Array
            N x 1 array of labels

    Returns:
        x : numpy.Array
            N_1 x k x 900 x D numpy array representing the xy coordinates of the first k periods of each worm

        y : numpy.Array
            N_1 x 1 array of labels 
    """


    return x[y>=threshold], y[y>=threshold], y>=threshold



    


def fill_na_interpolation(x):
    """
    Fills missing data using linear interpolation.

    Parameters:
        x : numpy.Array
            N x k x 900 x D numpy array representing the xy coordinates of the first k periods of each worm
    
    Returns:
        x : numpy.Array
            N x k x 900 x D numpy array representing the xy coordinates of the first k periods of each worm, with nan values filled by linear interpolation

    """
    x_filled = np.array([np.apply_along_axis(lambda m: pd.Series(m).interpolate(method='linear').ffill().bfill().to_numpy(), axis=1, arr=matrix) for matrix in x])
    return x_filled


