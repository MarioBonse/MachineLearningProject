#../venv/bin/python3
import numpy as np
import pandas as pd
import datacontrol
import sklearn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import validation
from sklearn.linear_model import LinearRegression

def main():
    try:
        TrainingData = datacontrol.readFile("data/ML-CUP18-TR.csv")
    except:
        TrainingData = datacontrol.readFile("../data/ML-CUP18-TR.csv")
    x_train, y_train = datacontrol.divide(TrainingData)
    # preprocessing
    #x_train = sklearn.preprocessing.scale(x_train, axis=0, with_mean=True, with_std=True, copy=True)
    input_dimention = x_train.shape[1]
    output_dimention = y_train.shape[1]
    # Now we will use k_fold in order to validate the model
    kf = KFold(n_splits=5)
    # scaler for NN
    scaler = StandardScaler()
    resultVal = []
    resultTest = []
    for train, test in kf.split(x_train):
        startTime = time.time()
        X_train, x_test, Y_train, y_test = x_train[train], x_train[test], y_train[train], y_train[test]
        
        reg = LinearRegression().fit(X_train, Y_train)