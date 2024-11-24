import numpy as np
import pandas as pd
import os

def read_lifespan_file(filename):
    """
    Reads .csv data from a specific lifespan file and returns it as a 2D numpy array
    
    Parameters:
        filename : str
            The path of the .csv file

    Returns:
        x: numpy.Array
            C x D numpy array representing all features in a worm's .csv file
    """
    # read the .csv file into a pandas dataframe
    try:
        df = pd.read_csv(filename, header=None)
        print(f"Shape of file {filename}: {df.shape}")
        return df.to_numpy()

    except Exception as e:
        print(f"Error reading {filename}: {e}")

def load_lifespan_data(data_path):
    """
    Imports lifespan data from .csv files for use in a NN.

    Parameters
    ----------
        data_path : str
            The file path to the folder containing all lifespan data files.

    Returns
    -------
        x : np.Array            
            N x C x D matrix containing N worms, C feature categories, and D features values 
            per category, taken as each worm data matrix.        
        y : np.Array            
            N x 1 array of labels indicating whether a worm belongs to the control.
    """
    # list all treated and control .csv files in the directory
    drug_path = data_path + "\companyDrug"    
    drug_files = os.listdir(drug_path)
    drug_files = [f for f in drug_files if f.endswith(".csv")]

    control_path = data_path + "\control"
    control_files = os.listdir(control_path)
    control_files = [f for f in control_files if f.endswith(".csv")]

    # populate x matrices with file data
    drug_x = []
    control_x = []

    for file in os.listdir(drug_path):
        if file.endswith(".csv"):
            filename = os.path.join(drug_path, file)
            drug_x.append(read_lifespan_file(filename))

    for file in os.listdir(control_path):
        if file.endswith(".csv"):
            filename = os.path.join(control_path, file)
            control_x.append(read_lifespan_file(filename))

    # initialize y arrays with 1 for drug, 0 for control
    drug_y = np.ones(np.shape(drug_x)[0])
    control_y = np.zeros(np.shape(control_x)[0])

    # combine drug and control x and y into one x and one y matrix
    x = np.concatenate(drug_y, control_x, axis=0)
    y = np.concatenate(control_y, control_x, axis=0)

    return x, y