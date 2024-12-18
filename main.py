# %matplotlib inline
import torch
import numpy as np
import argparse

import data_loading
import neural_net
import utils
from data_cleaning import *
from config import *
from modelContainer import modelContainer
import simple_models
from features import get_features

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-t", "--Task", help = "task to perform:\
                    \n  'clf': classification, using settings defined in config.py;\
                    \n  'reg': regression, using settings defined in config.py;\
                    \n  'clf_opt': grid search optimization on classification task;\
                    \n  'reg_opt': grid search optimization on regression task", default = "reg")
parser.add_argument("-m:", "--Model", help = "Model(s) to use:\
                    \n  'simple': simple models;\
                    \n  'CNN': convolutional neural network", default = "CNN")
parser.add_argument("-d:", "--Drug", help = "which drug to run the training: '1' or '2'", default = 1, type=int)
parser.add_argument("-s", "--SameGroup", help =  "run testing on same drug as training: 'True' or 'False'", default = "True" )
parser.add_argument("-r", "--Reproducible", help = "use reproducible results: 'True or 'False'", default = "True")
parser.add_argument("-l", "--LoadWeights", help = "load weights from a previous model: 'True' or 'False'", default = "False")
parser.add_argument("-w", "--WeightsPath", help = "path to weights file", default = None)
parser.add_argument("-c", "--saveWeights", help = "save weights to a file: 'True' or 'False'", default = "False")

# Read arguments from command line
args = parser.parse_args()

# Set seed to obtain reproducible results
if eval(args.Reproducible):
    torch.manual_seed(0)
    np.random.seed(0)

# MAIN DATA LOADING AND SPLITTING
if args.Task == "clf" or args.Task == "clf_opt":
    # main train/test split
    x_train, y_train, x_val, y_val, x_test_main, y_test_main = data_loading.load_clf_data(periods=8,drug=args.Drug, same_group=eval(args.SameGroup))
    y_test_drug_binary_main = None

elif args.Task == "reg" or args.Task == "reg_opt":
    # main train/test split
        x_train, y_train, x_val, y_val, x_test_main, y_test_main, y_test_drug_binary_main = data_loading.load_reg_data(death_threshold=2.5, death_length=1300,periods=8, same_group=eval(args.SameGroup),drug=args.Drug)

# OPTIMIZATION TASK (GRID-SEARCH)
if args.Task == "clf_opt" or args.Task == "reg_opt":

    # number of folds used in grid search
    fold_k = 4

    if args.Task == "clf_opt":
        grid_search_dict = GRID_SEARCH_CLF
        num_perf = 2 #number of performance metrics

    elif args.Task == "reg_opt":
        grid_search_dict = GRID_SEARCH_REG
        num_perf = 4 #number of performance metrics

    if eval(args.Reproducible):
        grid_search_dict["reproducible"] = [True]
    
    # get all combinations of hyperparameters to grid search
    grid_search_combinations = utils.get_combinations_hyperparameters(grid_search_dict)

    # recombine x_train and x_val (to be split again within k-fold later)
    x_train_main = torch.cat((x_train, x_val), dim=0)
    y_train_main = torch.cat((y_train, y_val), dim=0)

    best_performance = np.zeros(num_perf)
    best_hyperparameters_comb_idx = 0
    best_hyperparameters = grid_search_combinations[0]

    # writing a .csv for each hyperparameter combination performance
    with open(f"{args.Task}.csv", "w") as file:
        header = "index,"
        for key in grid_search_combinations[0].keys():
            header += key + ","
        header += "accuracy,f1-score\n" if args.Task == "clf_opt" else "mse,mae,rmse,r2,\n"
        file.write(header)

        # loop through hyperparameter combinations
        for comb_idx, combination in enumerate(grid_search_combinations):

            print(f"Running combination {comb_idx+1}/{len(grid_search_combinations)}.")

            # main train data split into k-folds
            x_kfold, y_kfold = data_loading.split_kfold(x_train_main, y_train_main, fold_k = fold_k)
        
            # track performance for each fold to average at the end
            model_performance = np.zeros((fold_k, num_perf))      

            # k-fold on given combination    
            for fold, (x_train, x_test) in enumerate(x_kfold):

                y_train = y_kfold[fold][0]
                y_test = y_kfold[fold][1]

                # further splitting x_train of given fold into train and validation sets
                if args.Task == "reg_opt":
                    x_train, x_val, y_train, y_val = data_loading.test_train_split_regression(x_train, y_train, 0.8)
                elif args.Task == "clf_opt":
                    x_train, x_val, y_train, y_val = data_loading.test_train_split_categorical(x_train, y_train, 0.8, equal_label_prop=True)

                x_train = x_train.permute(0,3,1,2)
                x_val = x_val.permute(0,3,1,2)
                x_test = x_test.permute(0,3,1,2)
                x_train = x_train.requires_grad_()
                x_val = x_val.requires_grad_()
                x_test = x_test.requires_grad_()
            
                # initialize new model
                kernels = [5] + [3]*combination["hidden_layer_L"] + [1]
                strides = [1] + [2]*combination["hidden_layer_L"] + [1]
                padding = [2] + [2]*combination["hidden_layer_L"] + [0]
                model = neural_net.CNN_model(2, 1, combination["hidden_layer_L"], combination["hidden_layer_K"], kernels, strides, padding, torch.nn.ReLU)

                # create model container object with combination of configuration arguments, run training and evaluation
                model_obj = modelContainer(model, **combination)
                model_obj.train(x_train, y_train, x_val, y_val)
                _, model_performance[fold] = model_obj.eval(args.Task, x_test, y_test, None)

            average_performance = np.average(model_performance, axis=0)

            # write performance to .csv file
            writeline = f"{comb_idx},"
            for value in combination.values():
                writeline += f"{value},"
            for metric in average_performance:
                writeline += f"{metric},"
            file.write(writeline + "\n")

            # check if performance is better than previous
            if args.Task == "clf_opt":
                if average_performance[0] > best_performance[0]: # comparing accuracy
                    best_performance = average_performance
                    best_hyperparameters_comb_idx = comb_idx
                    best_hyperparameters = combination

            elif args.Task == "reg_opt":
                if average_performance[1] < best_performance[1]: # comparing MAE
                    best_performance = average_performance
                    best_hyperparameters_comb_idx = comb_idx
                    best_hyperparameters = combination
            
        print(f"Optimal set of hyperparameters found at combination index {best_hyperparameters_comb_idx}")

    # training final model on optimal hyperparameters
    kernels = [5] + [3]*best_hyperparameters["hidden_layer_L"] + [1]
    strides = [1] + [2]*best_hyperparameters["hidden_layer_L"] + [1]
    padding = [2] + [2]*best_hyperparameters["hidden_layer_L"] + [0]
    model = neural_net.CNN_model(2, 1, best_hyperparameters["hidden_layer_L"], best_hyperparameters["hidden_layer_K"], kernels, strides, padding, torch.nn.ReLU)

    if args.Task == "reg_opt":
        x_train_main, x_val_main, y_train_main, y_val_main = data_loading.test_train_split_regression(x_train_main, y_train_main, 0.8)
    elif args.Task == "clf_opt":
        x_train_main, x_val_main, y_train_main, y_val_main = data_loading.test_train_split_categorical(x_train_main, y_train_main, 0.8, equal_label_prop=True)

    x_train_main = x_train_main.permute(0,3,1,2)
    x_val_main = x_val_main.permute(0,3,1,2)
    x_test_main = x_test_main.permute(0,3,1,2)
    x_train_main = x_train_main.requires_grad_()
    x_val_main = x_val_main.requires_grad_()
    x_test_main = x_test_main.requires_grad_()

    best_hyperparameters["verbose"] = True
    print("\nTraining model at optimal set of hyperparameters.")
    model_obj = modelContainer(model, **best_hyperparameters)
    model_obj.train(x_train_main, y_train_main, x_val_main, y_val_main)
    _, performance_main = model_obj.eval(args.Task, x_test_main, y_test_main, y_test_drug_binary_main)

# NON-OPTIMIZATION TASKS
else:
    if args.Model == "CNN":

        if args.Task == "clf":
            configs = CNN_CLF_CONFIG
        
        elif args.Task == "reg":
            configs = CNN_REG_CONFIG

        if eval(args.Reproducible):
            configs["reproducible"] = True

        x_train = x_train.permute(0,3,1,2)
        x_val = x_val.permute(0,3,1,2) 
        x_test = x_test_main.permute(0,3,1,2)
        x_train = x_train.requires_grad_()
        x_val = x_val.requires_grad_()    
        x_test = x_test.requires_grad_()

        # initialize new model
        kernels = [5] + [3]*configs["hidden_layer_L"] + [1]
        strides = [1] + [2]*configs["hidden_layer_L"] + [1]
        padding = [2] + [2]*configs["hidden_layer_L"] + [0]
        model = neural_net.CNN_model(2, 1, configs["hidden_layer_L"], configs["hidden_layer_K"], kernels, strides, padding, torch.nn.ReLU)

        model_obj = modelContainer(model, **configs)

        if eval(args.LoadWeights):
            model_obj.load_weights(args.WeightsPath)
        else:
            model_obj.train(x_train, y_train, x_val, y_val)

        if eval(args.saveWeights):
            model_obj.save_weights(args.WeightsPath)
        
        model_obj.eval(args.Task, x_test, y_test_main, y_test_drug_binary_main)

    elif args.Model == "simple":
        
        x_train = torch.cat([x_train, x_val],0)
        y_train = torch.cat([y_train, y_val],0)
        x_train = get_features(x_train)
        x_test = get_features(x_test_main)

        x_train = x_train.detach().numpy()
        y_train = y_train.detach().numpy()
        x_test = x_test.detach().numpy()
        y_test_main = y_test_main.detach().numpy()

        if args.Task == "clf":
            simple_models.clf_task(x_train, y_train, x_test, y_test_main)
        elif args.Task == "reg":
            simple_models.reg_task(x_train, y_train, x_test, y_test_main, y_test_drug_binary_main)
