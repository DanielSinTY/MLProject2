# %matplotlib inline
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

import matplotlib
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from functools import partial
from torchvision import datasets, transforms

import data_loading
import neural_net
from neural_net import MLP_model
import utils

import pandas as pd

from data_cleaning import *

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torcheval.metrics import R2Score

import random
pwd = os.getcwd()


class modelContainer():
    def __init__(self, model,**kwargs):
        """
        Initialize a model container object.
        
        Parameters:
            model : torch.nn.Module
                The model to train.
            
            optimizer : torch.optim
                The optimizer to use.
            
            criterion : torch.nn.Module
                The loss function to use.
            
            epochs : int
                The number of epochs to train the model.
            
            device : torch.device
                The device to train the model on.
            
            verbose : bool
                Whether to print training information.
            
            early_stopping : bool
                Whether to use early stopping.
            
            early_stopping_thresold : float
                The threshold for early stopping.
            
            threshold : float
                The decision threshold for the model.
        """
        
        self.model = model
        self.optimizer = kwargs.get("optimizer", torch.optim.Adam(model.parameters(), lr=kwargs.get("lr", 1e-4), weight_decay=kwargs.get("weight_decay", 1e-5)))
        self.criterion = kwargs.get("criterion", torch.nn.MSELoss())
        self.epochs = kwargs.get("epochs", 1000)
        self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.verbose = kwargs.get("verbose", True)
        self.early_stopping = kwargs.get("early_stopping", True)
        self.early_stopping_thresold = kwargs.get("early_stopping_thresold", 1.5)
        self.threshold = kwargs.get("threshold", None)
        if kwargs.get("reproducible", False):
            if self.verbose:
                print("Setting seeds for reproducible results.")
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
    def train(self, x_train, y_train, x_val, y_val):
        """
        Train the model.

        Parameters:
            x_train : torch.Tensor
                The training data.
            
            y_train : torch.Tensor
                The training labels.
            
            x_val : torch.Tensor
                The validation data.
            
            y_val : torch.Tensor
                The validation labels.
        """

        self.model = self.model.to(self.device)
        self.model.train()

        loss_tracker = []
        val_loss_tracker = []
        
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            output = self.model(x_train).view(-1)
            loss = self.criterion(output, y_train)
            loss.backward()
            self.optimizer.step()
            
            if self.verbose:
                
                self.model.eval()
                y_val_pred = self.model(x_val).view(-1)
                val_loss = self.criterion(y_val_pred, y_val)
                val_loss_tracker.append(val_loss.item())
                loss_tracker.append(loss.item())
                # line_val.set_xdata(range(0, len(val_loss_tracker)))
                # line_val.set_ydata(val_loss_tracker)
                self.model.train()
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {loss.item()} - Validation Loss: {val_loss.item()}")
            
            if self.early_stopping:
                # if the loss is greater than past 20 epochs, stop training
                if len(val_loss_tracker) > 50 :
                    if val_loss.item() > max(val_loss_tracker[-21:-1]):
                        if self.verbose:
                            print("Early stopping due to validation loss increase.")
                        break
                if loss.item() < self.early_stopping_thresold:
                    if self.verbose:
                        print("Early stopping due to validation loss below threshold.")
                    break
        
        if self.verbose:
            print("Training complete.")

            fig, ax = plt.subplots()
            ax.set_xlabel("Epoch", fontsize=20)
            ax.set_ylabel("Loss", fontsize=20)
            

            ax.plot(range(0, len(loss_tracker)), loss_tracker, label="Train Loss", linewidth=3)
            ax.plot(range(0, len(val_loss_tracker)), val_loss_tracker, label="Validation Loss", linewidth=3)
            ax.legend(fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.show()

    def calculate_performance(self, mode, y_pred, y_test, y_test_drug_binary=None):
        """
        Calculate the performance metrics of a prediction output compared to test labels.

        Parameters:
            mode : str
                The mode of the model. Either "reg"/"reg_opt" or "clf"/"clf_opt", which stands for regression and classification.
            y_pred : torch.Tensor
                The predicted labels.
            y_test : torch.Tensor
                The test labels.
            y_test_drug_binary : torch.Tensor
                The test labels for the drug worms.
        Returns:
            performance : array
                An array containing performance metrics.
                For mode='clf': [acc, f-score]
                For mode='reg': [mse, mae, rmse, r2]
        """
        if self.verbose:
            print("Test results:")
        
        if mode == "reg" or mode == "reg_opt":
            if y_test_drug_binary is not None:
                groups = ["overall", "drug", "control"]
            else:
                groups = ["overall"]
            for group in groups:
                if group == "drug":
                    mask = y_test_drug_binary
                elif group == "control":
                    mask = ~y_test_drug_binary
                else:
                    mask = np.ones(len(y_pred), dtype=bool)
                mse = torch.nn.MSELoss()(y_test[mask], y_pred[mask]).detach()
                mae = torch.nn.L1Loss()(y_test[mask], y_pred[mask]).detach()
                rmse = torch.sqrt(mse).detach()
                metric = R2Score()
                metric.update(y_test[mask], y_pred[mask])
                r2 = metric.compute().detach()
                if self.verbose:
                    print(f"Predicted labels: {y_pred[mask]}")
                    print(f"Predicted labels: {y_test[mask]}")
                    print(f"{group.capitalize()} - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

            return [mse, mae, rmse, r2]
        elif mode == "clf" or mode == "clf_opt":
            probs = torch.sigmoid(y_pred)
            probs = probs.detach().numpy()

            y_pred = (probs > self.threshold).astype(int)
            y_test = y_test.detach().numpy()
            acc = utils.compute_accuracy_classification(y_test, y_pred)
            f_score = utils.compute_fscore_classification(y_test, y_pred)

            if self.verbose:
                print(f"Predicted labels: {y_pred}")
                print(f"Predicted labels: {y_test}")
                print(f"Accuracy: {acc}")
                print(f"F-score: {f_score}")

            return [acc, f_score]

    def eval(self, mode, x_test, y_test,y_test_drug_binary=None):
        """
        Evaluate the model.

        Parameters:
            mode : str
                The mode of the model. Either "reg" or "clf", which stands for regression and classification.
            x_test : torch.Tensor
                The test data.
            
            y_test : torch.Tensor
                The test labels.

            y_test_drug_binary : torch.Tensor
                The test labels for the drug worms.

        Returns:
            y_pred : torch.Tensor
                The predicted test labels.

            performance : numpy.Array
                A horizontal array containing performance metrics.
                For mode='clf': [acc, f-score]
                For mode='reg': [mse, mae, rmse, r2]
        """
        
        self.model.eval()
        y_pred = self.model(x_test).view(-1)

        performance = self.calculate_performance(mode, y_pred, y_test, y_test_drug_binary)
        performance = np.array(performance)

        return y_pred, performance
    
    def save_weights(self, path):
        """
        Save the model weights.

        Parameters:
            path : str
                The path to save the model weights.
        """
        if path is None:
            path = "weights.pth"
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path):
        """
        Load the model weights.

        Parameters:
            path : str
                The path to load the model weights.
        """
        if path is None:
            path = "weights.pth"


        
        self.model.load_state_dict(torch.load(path))
        self.model.eval()