import torch

DEATH_THRESHOLD = 2.5
DEATH_LENGTH = 1300

CNN_CLF_CONFIG = {
    "lr" : 1e-4,
    "weight_decay" : 1e-5,
    "epochs": 500,
    "threshold" : 0.4, # Decision threshold between null and unity label ([0,1])
    "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "verbose" : True,
    "early_stopping" : False,
    "early_stopping_thresold" : None,
    "criterion" : torch.nn.BCEWithLogitsLoss(),    
    "hidden_layer_K" : 16,
    "hidden_layer_L" : 1,
}

CNN_REG_CONFIG = {
    "lr" : 1e-5,
    "weight_decay" : 1e-5,
    "epochs": 500,
    "threshold" : None, # Decision threshold between null and unity
    "device" : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "verbose" : True,
    "early_stopping" : True,
    "early_stopping_thresold" : 0.5,
    "criterion" : torch.nn.MSELoss(),  
    "hidden_layer_K" : 64,
    "hidden_layer_L" : 1,
}

GRID_SEARCH_REG = {
    "lr" : [1e-5, 1e-4, 1e-3],
    "weight_decay" : [1e-5, 1e-4, 1e-3],
    "epochs": [500, 1500],
    "threshold" : [None], # Decision threshold between null and unity
    "device" : [torch.device("cuda" if torch.cuda.is_available() else "cpu")],
    "verbose" : [False],
    "early_stopping" : [True, False],
    "early_stopping_thresold" : [0.5],
    "criterion" : [torch.nn.MSELoss()],
    "hidden_layer_K" : [16, 32, 64],
    "hidden_layer_L" : [1, 2]
}

GRID_SEARCH_CLF = {

    "lr" : [1e-5, 1e-4, 1e-3],
    "weight_decay" : [1e-5, 1e-4, 1e-3],
    "epochs" : [100, 500],
    "threshold" : [0.4, 0.5, 0.6],
    "device" : [torch.device("cuda" if torch.cuda.is_available() else "cpu")],
    "verbose" : [False],
    "early_stopping" : [False],
    "early_stopping_thresold" : [None],
    "criterion" : [torch.nn.BCEWithLogitsLoss()],    
    "hidden_layer_K" : [16, 32, 64],
    "hidden_layer_L" : [1, 2]
}