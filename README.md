# CS 433 Machine Learning Project 2: Drug Treatment Classification and Lifespan Prediction of C. elegans using Machine Learning Techniques

## Overview
This repository contains the code of CS 433 Machine Learning Project 2. In this project, we utilise different machine learning models to disguish c. elegans treated with drugs from those not and predict lifespan of c. elegans, 
using early behavioral data, in specific, x-y coordinates of center of mass in the first 2 days.

Note: As the data set is small, we decided to include it in this repository. Users can directly run the code without extra steps to download the data

## Installation
To get started, clone this repository:
```
git clone https://github.com/DanielSinTY/MLProject2.git
cd MLProject2
```
and install the dependencies:
```
pip install requirements.txt
```

## Usage
To run the code, run
```
python main.py
```
This command contains various arguments to be set by the users, for different tasks of the project, details of the arguments can be found by running
```
python main.py --help
```

## Repository Structure
`main.py` contains the main file to be run for the project

`data_loading.py` contains functions to load the data into desired format

`data_cleaning.py` contains helper functions to clean the data, by filling nan values and remove data with excessive nan values

`features.py` generates features from the raw data

`simple_models.py` contains simple models from scikit-learn with default settings, and helper funcitons to train and evaluate the models on the project tasks.

`neural_net.py` is the model factory which generates the CNN model.

`modelContainer.py` is a model container object that contains a CNN model, and helper functions for training and evaluation for the project tasks.

`utils.py` contains other helper functions

`data/Lifespan` contains all the data for this project
