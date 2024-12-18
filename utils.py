import numpy as np
from itertools import product

def compute_accuracy_classification(y_true, y_pred):
    """
    Computes the accuracy score by comparing an array of predicted labels to the array of actual labels.
    
    Parameters:
        y_true (np.Array) : The array of actual labels, with values in {0,1}.
        y_pred (np.Array) : The array of predicted labels, with values in {0,1}.

    Returns:
        acc (float) : The accuracy score of the prediction, with values in [0,1].
    """
    # Validate inputs
    if y_pred.shape != y_true.shape:
        raise ValueError("Predicted and true label arrays must have the same shape.")

    if not (np.isin(y_pred, [0, 1]).all() and np.isin(y_true, [0, 1]).all()):
        raise ValueError("Both arrays must contain only binary values (0 or 1).")

    # Compute accuracy
    correct_predictions = np.sum(y_pred == y_true)
    total_predictions = len(y_true)

    return correct_predictions / total_predictions

def compute_fscore_classification(y_true, y_pred):
    """
    Computes the F-score by comparing an array of predicted labels to the array of actual labels.
    
    Parameters:
        y_true (np.Array) : The array of actual labels, with values in {0,1}.
        y_pred (np.Array) : The array of predicted labels, with values in {0,1}.

    Returns:
        f_score (float) : The F-score  of the prediction, with values in [0,1].
    """
    # Validate inputs
    if y_pred.shape != y_true.shape:
        raise ValueError("Predicted and true label arrays must have the same shape.")

    if not (np.isin(y_pred, [0, 1]).all() and np.isin(y_true, [0, 1]).all()):
        raise ValueError("Both arrays must contain only binary values (0 or 1).")

    # Compute precision, recall, and F1 score
    true_positive = np.sum((y_pred == 1) & (y_true == 1))
    false_positive = np.sum((y_pred == 1) & (y_true == 0))
    false_negative = np.sum((y_pred == 0) & (y_true == 1))

    if true_positive == 0:
        return 0.0  # Avoid division by zero

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    return 2 * (precision * recall) / (precision + recall)

def get_combinations_hyperparameters(config_dict):
    """
    Takes a dictionary of arrays containing hyperparameter values, and returns an array of dictionaries with all the
    combinations of those hyperparameters.
    Parameters:
        config_dict (dict) : A dictionary of arrays for hyperparameter options.
    Returns:
        grid_search_combinations (array) : An array of all the combinations of hyperparameters stored in
        dictionaries, accessible by their original keywords.
    """
    # Extract keys and values from the dictionary
    keys = config_dict.keys()
    values = config_dict.values()
    
    # Generate all combinations of the hyperparameter values
    combinations = list(product(*values))
    
    # Convert combinations into dictionaries with the corresponding keys
    grid_search_combinations = [dict(zip(keys, combination)) for combination in combinations]
    
    return grid_search_combinations