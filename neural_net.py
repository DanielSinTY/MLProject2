import torch
import numpy as np
import torch.nn as nn


class MLP_model(nn.Module):
    """
    Class for a multilayer perceptron with linear layers.
    """
    def __init__(self, D, O, K, L, phi=nn.ReLU):
        """
        Initialize a feed-forward neural network with variable input and output layer width,
        number and width of hidden layers, and activation function.

        Parameters:
            D (int) : The width of the input layer.
            O (int) : The width of the output layer.
            K (int) : The width of the hidden layers.
            L (int) : The number of hidden layers.
            phi (torch.nn.Module) : The activation function used between layers. Default = ReLU.
        """
        super(MLP_model, self).__init__()

        layers = []

        # input layer
        layers.append(nn.Linear(D,K))
        layers.append(phi())

        # hidden layers
        for i in range(L):
            layers.append(phi())
            layers.append(nn.Linear(K, K))

        # output layer
        layers.append(nn.Linear(K, O))

        # combine layers
        self.model = nn.Sequential(*layers) 

    def forward(self, Z):
        """
        Forward pass with a MLP.
        
        Parameters
            Z (torch.Tensor) : A torch tensor with the input data.
        Returns
            model (CNN_model) : The model after a forward pass.
        """
        return self.model(Z)

class CNN_model(nn.Module):
    """
    Class for a convolutional neural net with 2D convolution layers.
    """
    def __init__(self, D, O, L, K, kernels, strides, padding, phi=nn.ReLU):
        """
        Initialize a convolutional neural network with variable input and output layer width,
        number and width of hidden layers, custom strides and padding, and activation function.

        Parameters:
            D (int) : The width of the input layer.
            O (int) : The width of the output layer.
            L (int) : The number of hidden layers.
            K (int or array) : A L+1-length array of integers containing the widths of the hidden layers. If just
            an integer, the uniform width of all hidden layers.
            kernels (int or array): A L+2-length array of integers or tuples for each layer's kernel values. If
            just an integer or tuple, the uniform kernels of all layers.      
            strides (int or array) : A L+2-length array of integers or tuples for each layer's stride values. If
            just an integer, the uniform strides of all layers.
            padding (int or array) : A L+2-length array of integers or tuples for each layer's padding values. If
            just an integer, the uniform padding of all layers.
            phi (torch.nn.Module) : The activation function used between layers. Default = ReLU.
        """
        super(CNN_model, self).__init__()

        assert isinstance(K, int) or isinstance(K, tuple) or isinstance(K, list) and len(K) == L+1, "K must be an integer or a list of length L"
        assert isinstance(kernels, int) or isinstance(kernels, tuple) or isinstance(kernels, list) and len(kernels) == L+2, "kernels must be an integer, a tuple, or a list of length L+2"
        assert isinstance(strides, int) or isinstance(strides, tuple) or isinstance(strides, list) and len(strides) == L+2, "strides must be an integer, a tuple or a list of length L+2"
        assert isinstance(padding, int) or isinstance(padding, tuple) or isinstance(padding, list) and len(padding) == L+2, "padding must be an integer, a tuple or a list of length L+2"
        
        layers = []

        # input layer
        K_in = K[0] if type(K) == list else K
        kernels_in = kernels[0] if type(kernels) == list else kernels
        strides_in = strides[0] if type(strides) == list else strides
        padding_in = padding[0] if type(padding) == list else padding
            
        layers.append(nn.Conv2d(D, K_in, kernels_in, strides_in, padding_in))
        layers.append(phi())

        # hidden layers
        for i in range(L):
            K_i = K[i] if type(K) == list else K
            K_o = K[i+1] if type(K) == list else K
            kernels_i = kernels[i+1] if type(kernels) == list else kernels
            strides_i = strides[i+1] if type(strides) == list else strides
            padding_i = padding[i+1] if type(padding) == list else padding

            
            #print(f"in_channels = {K_i}, out_channels = {K_o}")
            layers.append(nn.Conv2d(K_i, K_o, kernels_i, strides_i, padding_i))
            layers.append(phi())

        # output layer
        kernels_o = kernels[-1] if type(kernels) == list else kernels
        strides_o = strides[-1] if type(strides) == list else strides
        padding_o = padding[-1] if type(padding) == list else padding
        
        K_i = K[-1] if type(K) == list else K
        layers.append(torch.nn.AdaptiveAvgPool2d(1))
        layers.append(torch.nn.Conv2d(K_i, O, kernels_o, strides_o, padding_o))
        layers.append(torch.nn.Flatten())

        # combine layers
        self.model = nn.Sequential(*layers)

    def forward(self, Z):
        """
        Forward pass with a CNN.
        
        Parameters
            Z (torch.Tensor) : A torch tensor with the input data.
        Returns
            model (CNN_model) : The model after a forward pass.
        """
        return self.model(Z)