import numpy as np
import pandas as pd
import os
from config import *
from data_cleaning import *
import torch
from sklearn.model_selection import KFold

def read_lifespan_file(filename):
    """
    Reads .csv data from a specific lifespan file and returns it as a 2D numpy array
    
    Parameters:
        filename (str)
            The path of the .csv file

    Returns:
        x (numpy.Array)
            C x D numpy array representing all features in a worm's .csv file
    """
    # read the .csv file into a pandas dataframe
    try:
        # remove header row, convert all to float, replace missing values with -1
        df = pd.read_csv(filename, header=0, skiprows=1, dtype = np.float64)
        # df = df.fillna(-1.)

        name = filename.rsplit("\\", 1)[-1]
        #print(f"Shape of file {name}: {df.shape}")
        return df.to_numpy()

    except Exception as e:
        print(f"Error reading {filename}: {e}")

def load_lifespan_data(data_path,mode="binary", drug_path=None, control_path=None, DEATH_THRESHOLD= DEATH_THRESHOLD, DEATH_LENGTH=DEATH_LENGTH, find_optimal_death_threshold=False, configList=None):
    """
    Imports lifespan data from .csv files for use in a NN.

    Parameters
    ----------
        data_path (str) : The file path to the folder containing all lifespan data files.
        mode (str) : The type of output y, "binary" for whether the worm is drugged, "lifespan" for lifespan of the worm

    Returns
    -------
        x (list of numpy.Array)
            N x C x D matrix containing N worms, C feature categories, and D features values 
            per category, taken as each worm data matrix. D varies by file, so needs further 
            processing before being usable as a 3D matrix.
        y (numpy.Array)
            N x 1 array of labels
    """
    if drug_path is None:
        drug_path = data_path + "\companyDrug"
    else:
        drug_path = data_path + drug_path
    if control_path is None:
        control_path = data_path + "\control"
    else:
        control_path = data_path + control_path
    # list all treated and control .csv files in the directory
    # drug_path = data_path + "\companyDrug"    
    drug_files = os.listdir(drug_path)
    drug_files = [f for f in drug_files if f.endswith(".csv")]

    # control_path = data_path + "\control"
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

    # combine drug and control x and y into one x and one y matrix
    x = drug_x + control_x
    num_drug = len(drug_x)
    num_control = len(control_x)

    

    if mode == 'lifespan':
        if find_optimal_death_threshold:   
            thresholdList = [0.1,0.2,0.5,0.75,1,1.5,2,2.5,5,8,10,15]
            deathLengthList = [50,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1800,2100]
            avg_drug_lifespan = []
            avg_control_lifespan = []
            diff = []
            
        else:
            thresholdList = [DEATH_THRESHOLD]
            deathLengthList = [DEATH_LENGTH]
        if not configList:
            configList = [(threshold, deathLength) for threshold in thresholdList for deathLength in deathLengthList]
        y = np.zeros(len(x))
        for deathThreshold, deathLength in configList:
            for i in range(len(x)):
                worm_xy = x[i][:,[2,3]]
                delta_xy = np.diff(worm_xy, axis=0)
                speed = np.linalg.norm(delta_xy, axis=1)
                inactive_frames = speed<deathThreshold
                cum_inactive_frames = np.cumsum(inactive_frames, axis=0)
                consecu_inactive = cum_inactive_frames-np.roll(cum_inactive_frames, deathLength, axis=0)
                dead = consecu_inactive==(deathLength-1)
                res = np.where(dead)[0]
                if res.shape[0] == 0:
                    y[i] = worm_xy.shape[0]//900/4
                else:
                    y[i] = res[0]//900/4
            if find_optimal_death_threshold:
                if np.isnan(y).sum() > 0 :
                    print("Threshold: ", deathThreshold, "Death Length: ", deathLength)
                    print(y)
                    continue
                drug_avg = np.nanmean(y[:num_drug])
                control_avg = np.nanmean(y[num_drug:])
                avg_drug_lifespan.append(drug_avg)
                avg_control_lifespan.append(control_avg)
                diff.append(drug_avg-control_avg)

    else:
        # initialize y arrays with 1 for drug, 0 for control
        drug_y = np.ones(len(drug_x))
        control_y = np.zeros(len(control_x))
        y = np.concatenate((drug_y, control_y), axis=0, dtype=np.float64)

    if find_optimal_death_threshold:
        return x, y,configList, avg_drug_lifespan, avg_control_lifespan, diff

    return x, y

def test_train_split_categorical(x, y, split, equal_label_prop=False, seed=0, returnMask=False):
    """
    Take values and binary categorical labels matrices and splits it into a test and training set randomly.
    Accepts either numpy arrays or pytorch tensors, but must have consistent data type.

    Parameters
    ----------
        x (numpy.Array or pytorch.Tensor)
            The array of data points, after trimming.
        y (numpy.Array or pytorch.Tensor)
            The array of labels.
        split (float)
            The split fraction of all data points taken for training.
            Takes values between 0 and 1.
        equal_label_prop (bool)
            If true, forces the selection of random data points for training to be equally distributed
            between both label categories. Default = False.
        seed (float)
            Seed for random selection. Default = 0.

    Returns
    -------
        x_train (numpy.Array or pytorch.Tensor): Training data points.
        x_test (numpy.Array or pytorch.Tensor): Testing data points.
        y_train (numpy.Array or pytorch.Tensor): Training labels.
        y_test (numpy.Array or pytorch.Tensor): Testing labels.
    """
    if seed is not None:
        np.random.seed(seed)

    return_tensor = False
    if type(x) == torch.Tensor and type(y) == torch.Tensor:
        return_tensor = True
        x = x.numpy()
        y = y.numpy()

    if equal_label_prop:
        # Separate data by labels
        data_0 = x[y == 0]
        data_1 = x[y == 1]

        # Shuffle each group
        np.random.shuffle(data_0)
        np.random.shuffle(data_1)

        # Calculate the balanced training size for each group
        train_size = int(split * len(x) / 2)
        train_data_0 = data_0[:train_size]
        train_data_1 = data_1[:train_size]

        # Concatenate balanced training data and shuffle
        x_train = np.concatenate([train_data_0, train_data_1])
        y_train = np.concatenate([np.zeros(len(train_data_0)), np.ones(len(train_data_1))])

        shuffled_indices = np.random.permutation(len(x_train))
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]

        # Use the remaining data for testing
        x_test = np.concatenate([data_0[train_size:], data_1[train_size:]])
        y_test = np.concatenate([np.zeros(len(data_0[train_size:])),np.ones(len(data_1[train_size:]))])
        indices = np.arange(len(x_test))
        np.random.shuffle(indices)
        x_test = x_test[indices]
        y_test = y_test[indices]
    else:
        # Shuffle the entire dataset
        indices = np.arange(len(x))
        np.random.shuffle(indices)

        x = x[indices]
        y = y[indices]

        # Split based on the split fraction
        train_size = int(split * len(x))
        x_train = x[:train_size]        
        x_test = x[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]

    if return_tensor:
        x_train = torch.tensor(x_train)
        x_test = torch.tensor(x_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)

    if returnMask:
        return x_train, x_test, y_train, y_test, indices, train_size
    return x_train, x_test, y_train, y_test

def test_train_split_regression(x, y, split, seed=0):
    """
    Take values and regression labels matrices and splits it into a test and training set randomly.

    Parameters
    ----------
        x (numpy.Array or pytorch.Tensor)
            The array of data points, after trimming.
        y (numpy.Array or pytorch.Tensor)
            The array of labels.
        split (float)
            The split fraction of all data points taken for training.
            Takes values between 0 and 1.
        equal_label_prop (bool)
        seed (float)
            Seed for random selection. Default = 0.

    Returns
    -------
        x_train (numpy.Array or pytorch.Tensor): Training data points.
        x_test (numpy.Array or pytorch.Tensor): Testing data points.
        y_train (numpy.Array or pytorch.Tensor): Training labels.
        y_test (numpy.Array or pytorch.Tensor): Testing labels.
    """
    if seed is not None:
        np.random.seed(seed)

    return_tensor = False
    if type(x) == torch.Tensor and type(y) == torch.Tensor:
        return_tensor = True
        x = x.numpy()
        y = y.numpy()

    # Shuffle the entire dataset
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    x = x[indices]
    y = y[indices]

    # Split based on the split fraction
    train_size = int(split * len(x))
    x_train = x[:train_size]        
    x_test = x[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]

    if return_tensor:
        x_train = torch.tensor(x_train)
        x_test = torch.tensor(x_test)
        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)

    return x_train, x_test, y_train, y_test

def lifespan_time_window_split(x, y, window_length):
    """
    Takes lifespan data and reorganizes it such that data points are time windows of a specified
    length with less features per data point. Labels are changed from the total lifespan to
    the estimated time until death (TUD), given the worm behaviour, treatment, and time since
    birth.

    Requires data which was imported with "load_lifespan_data" using mode='lifespan'.

    Parameters
    ----------
        x (list of numpy.Array)
            N x C x D matrix containing N worms, C feature categories, and D features values 
            per category, taken as each worm data matrix.
        y (numpy.Array)    
            N x 1 array of labels indicating a worm's lifespan in hours.
        window_length (int)
            An integer indicating the number of frames to use within a window.
        
    Returns
    -------
        x_window (numpy.Array)
            Array of data points and features, with added label for time since birth.
        x_timealive_window (numpy.Array)
            N x 1 Array containing the total alive time of the worm by that window's reference frame, to be added
            to the labels tensor during flattening
        y_window (numpy.Array)
            N x 1 array of data labels adjusted to compute time until death (days) starting 
            from the start of the time window captured in a data point.
    """

    # For each worm, split into array of np.Array by video
        # split each video by window
        # for each window, calculate time since birth from first frame
        # and time until death

    assert window_length <= 900, "Error: window_length exceeds length of a video (900 frames)"
    
    x_window = []
    x_extralabels_window = []
    y_window = []

    # split into individual videos
    for i, worm in enumerate(x):
        N_videos = int(np.shape(worm)[0]/900) # number of complete videos
        split_indices = range(900, 900*(N_videos+1), 900)
        worm_split = np.split(worm, split_indices)

        # Calculate time until death for each video
        time_alive = np.arange(0, 0.25*(N_videos+1), 0.25)
        worm_lifespan = y[i]
        time_until_death = worm_lifespan * np.ones(np.shape(time_alive)) - time_alive

        # split each video into windows
        for j, video in enumerate(worm_split):
            trimmed = False
            N_windows = int(np.shape(video)[0]/window_length) # number of complete windows
            window_indices = range(window_length, window_length*(N_windows+1), window_length)
            windows = np.split(video, window_indices, axis=0)

            if len(windows[-1]) < window_length: 
                windows = windows[:-1]
                trimmed=True

            for window in windows: x_window.append(window)

            # Determine whether to trim a window from the labels array
            if trimmed==True: N_windows -= 1

            # Calculate time alive and time until death for each window
            window_duration = 2*window_length/60/60/24

            time_alive_window = window_duration * np.arange(0, window_duration*(N_windows+1), window_duration)
            time_alive_window = time_alive[j] * np.ones(np.shape(time_alive_window)) + time_alive_window

            for window in time_alive_window: x_extralabels_window.append(window)

            time_until_death_window = time_until_death[j] * np.ones(np.shape(time_alive_window)) - time_alive_window
            for window in time_until_death_window: y_window.append(window)

    # Convert to numpy arrays
    x_window = np.array(x_window)
    x_extralabels_window = np.array(x_extralabels_window)
    y_window = np.array(y_window)           

    return x_window, x_extralabels_window, y_window

def load_data(death_threshold=DEATH_THRESHOLD, death_length=DEATH_LENGTH,periods=8, drug=1, same_group=True, mode = "lifespan"):
    """
    Load data.

    Parameters
    ----------
        death_threshold (float) : The threshold speed below which a worm is considered dead.
        death_length (int) : The number of frames a worm must be below the death threshold to be considered dead.
        periods (int) : The number of periods to return
        drug (int) : Choose which drug data to load.
        same_group (bool) : Choose whether to set the test data as a split from the same drug's data, or to set it
        as the data for the other drug.

    Returns
    -------
        x_train_tensor (torch.Tensor) : Training data points.
        y_train_tensor (torch.Tensor) : Training labels.
        x_val_tensor (torch.Tensor) : Validation data points.
        y_val_tensor (torch.Tensor) : Validation labels.
        x_test_tensor (torch.Tensor) : Testing data points.
        y_test_tensor (torch.Tensor) : Testing labels.
        y_test_drug_binary (numpy.Array) : Testing labels for drug worms of whether they are drugged.
    """
    pwd = os.getcwd()

    # choosing between drug data
    if drug == 1:
        drug_path = "\\1_drug"
        control_path = "\\1_control"
    elif drug == 2:
        drug_path = "\\2_drug"
        control_path = "\\2_control"

    

    # loading data            
    x_init, y = load_lifespan_data(pwd + "\\data\\Lifespan", mode = mode,drug_path = drug_path, control_path = control_path, DEATH_LENGTH=death_length, DEATH_THRESHOLD=death_threshold)

    #data cleaning

    x_trimmed = get_first_n_periods(x_init, periods)
    x_trimmed, y, mask1 = remove_na(0.2, x_trimmed, y)
    if mode == "lifespan":
        x_trimmed, y, mask2 = remove_outliers(5, x_trimmed, y)
    x_trimmed_filled = fill_na_interpolation(x_trimmed)
    x_trimmed_filled, y, mask3 = remove_na(0.01, x_trimmed_filled, y)

    #print(f"Shape of x_trimmed matrix: {np.shape(x_trimmed)}")

    # Get train and test sets
    split = 0.7
    if mode == "lifespan":
        equal_label_prop = False
    else:
        equal_label_prop = True
    x_train, x_val, y_train, y_val,indices,train_size = test_train_split_categorical(x_trimmed_filled, y, split, equal_label_prop=equal_label_prop, returnMask=True)

    
    if same_group:

        # train_test_split = 0.7
        # x_train, x_test, y_train, y_test, indices,train_size = test_train_split_categorical(x_test_trimmed_filled, y_test, train_test_split, returnMask=True)

        x_test, y_test = x_val,y_val
        if mode == "lifespan":
            _,y_test_drug_binary = load_lifespan_data(pwd + "\\data\\Lifespan", drug_path = drug_path, control_path = control_path)
            y_test_drug_binary = y_test_drug_binary[mask1][mask2][mask3].astype(bool)
            y_test_drug_binary = y_test_drug_binary[indices][train_size:]
        train_val_split = 0.8
        
        if mode == "lifespan":
            equal_label_prop = False
        else:
            equal_label_prop = True
        x_train, x_val, y_train, y_val = test_train_split_categorical(x_train, y_train, train_val_split, equal_label_prop = equal_label_prop)

        
        x_train_tensor = torch.from_numpy(x_train).float()
        x_val_tensor = torch.from_numpy(x_val).float()
        x_test_tensor = torch.from_numpy(x_test).float()

        # x_train_tensor = x_train_tensor.requires_grad_()
        # x_val_tensor = x_val_tensor.requires_grad_()
        # x_test_tensor = x_test_tensor.requires_grad_()

        y_train_tensor = torch.from_numpy(y_train).float()
        y_val_tensor = torch.from_numpy(y_val).float()
        y_test_tensor = torch.from_numpy(y_test).float()
        if mode=="lifespan":
            return x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor, x_test_tensor, y_test_tensor, y_test_drug_binary
        else:
            return x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor, x_test_tensor, y_test_tensor
    

    if drug == 1:
        test_drug_path = "\\2_drug"
        test_control_path = "\\2_control"
    elif drug == 2:
        test_drug_path = "\\1_drug"
        test_control_path = "\\1_control"

    

    # x_train_flattened = np.array([x_trimmed_point.flatten() for x_trimmed_point in x_train], dtype=np.float32)
    # x_test_flattened = np.array([x_trimmed_point.flatten() for x_trimmed_point in x_test], dtype=np.float32)
    #print(f"Shape of x_trimmed matrix: {np.shape(x_flattened)}")
    x_test_init, y_test = load_lifespan_data(pwd + "\\data\\Lifespan", mode = mode, drug_path = test_drug_path, control_path = test_control_path, DEATH_LENGTH=death_length, DEATH_THRESHOLD=death_threshold)


    x_test_trimmed = get_first_n_periods(x_test_init, periods)
    x_test_trimmed, y_test,mask1 = remove_na(0.3, x_test_trimmed, y_test)
    if mode == "lifespan":
        x_test_trimmed, y_test,mask2 = remove_outliers(5, x_test_trimmed, y_test)
    x_test_trimmed_filled = fill_na_interpolation(x_test_trimmed)
    x_test_trimmed_filled,y_test,mask3 = remove_na(0.01, x_test_trimmed_filled, y_test)

    if mode == "lifespan":
        _,y_test_drug_binary = load_lifespan_data(pwd + "\\data\\Lifespan", drug_path = test_drug_path, control_path = test_control_path)
        y_test_drug_binary = y_test_drug_binary[mask1][mask2][mask3].astype(bool)

    # Convert numpy array into pytorch tensor
    x_train_tensor = torch.from_numpy(x_train).float()
    x_val_tensor = torch.from_numpy(x_val).float()

    # x_train_tensor = x_train_tensor.requires_grad_()
    # x_val_tensor = x_val_tensor.requires_grad_()

    y_train_tensor = torch.from_numpy(y_train).float()
    y_val_tensor = torch.from_numpy(y_val).float()

    x_test_tensor = torch.from_numpy(x_test_trimmed_filled).float()
    # x_test_tensor = x_test_tensor.requires_grad_()

    y_test_tensor = torch.from_numpy(y_test).float()


    
    if mode=="lifespan":
        return x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor, x_test_tensor, y_test_tensor, y_test_drug_binary
    else:
        return x_train_tensor, y_train_tensor, x_val_tensor, y_val_tensor, x_test_tensor, y_test_tensor

def load_clf_data(periods=8, drug=1, same_group=True):
   return load_data(periods=periods,drug=drug,same_group=same_group, mode="binary")

def load_reg_data(death_threshold=DEATH_THRESHOLD, death_length=DEATH_LENGTH,periods=8, drug=1, same_group=True):
    return load_data(death_threshold=death_threshold, death_length=death_length,periods=periods, drug=drug, same_group=same_group, mode="lifespan")

def split_kfold(x, y, fold_k=4):
    """
    Split pre-cleaned pytorch tensors into folds.
    Parameters:
        x (pytorch.Tensor) : Contains the data to be split into train and test folds.
        y (pytorch.Tensor) : Contains the labels to be split into train and test folds.
        fold_k (int) : Number of folds.
    Returns:
        x_kfold (array) : A fold_k-length array containing tuples with the train and validation data for a given
        fold as torch.Tensor objects.
        y_kfold (array) : A fold_k-length array containing tuples with the train and validation labels for a given
        fold as torch.Tensor objects.
    """
    x_kfold = [None]*fold_k
    y_kfold = [None]*fold_k
    kf = KFold(fold_k, shuffle=True)
    kf_indices = kf.split(x)
    for fold, (train_idx, test_idx) in enumerate(kf_indices):

        x_train = x[np.array(train_idx)]
        x_test = x[np.array(test_idx)]

        y_train = y[np.array(train_idx)]
        y_test = y[np.array(test_idx)]

        x_kfold[fold] = (x_train.float(), x_test.float())
        y_kfold[fold] = (y_train.float(), y_test.float())

    return x_kfold, y_kfold