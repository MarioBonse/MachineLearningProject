#!/usr/bin/python3
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
import torch

NetworArchitecture = [100, 10000, 1000, 100, 100]
activation = "relu"


def newModel(D_in, D_out):
    return torch.nn.Sequential(
        torch.nn.Linear(D_in, NetworArchitecture[0]),
        torch.nn.ReLU(),
        torch.nn.Linear(NetworArchitecture[0], NetworArchitecture[1]),
        torch.nn.ReLU(),
        torch.nn.Linear(NetworArchitecture[1], D_out),
    )

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
    result = []
    for train, test in kf.split(x_train):
        model = newModel(input_dimention, output_dimention)
        X_train, x_test, Y_train, y_test = x_train[train], x_train[test], y_train[train], y_train[test]
        # Now we will sclae the data
        # We will fit the scaler with the training set and apply the trasformation also
        # to the test data
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        x_test = scaler.transform(x_test)
        NN_output = []
        for x in range(np.shape(X_train)[0]):
            out = model(torch.from_numpy(X_train[x]).float())
            out = out.detach().numpy()
            NN_output.append(out)
        NN_output = np.array(NN_output)
        reg = LinearRegression().fit(NN_output, Y_train)
        #reg = MultiOutputRegressor(LinearRegression(), n_jobs=2).fit(NN_output, Y_train)
        score = 0
        reg = LinearRegression().fit(NN_output, Y_train[:, 0])
        #reg = MultiOutputRegressor(LinearRegression(), n_jobs=2).fit(NN_output, Y_train)
        score = 0
        for i in range(np.shape(x_test)[0]):
            out = model(torch.from_numpy(x_test[i]).float())
            out = out.detach().numpy()
            yout = reg.predict(np.asmatrix(out))
            resultmoment = (yout - y_test[i, 0])**2
            score += resultmoment
        score = score/np.shape(x_test)[0]
        result.append(score)
        #loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
    print("Mean: %.2f +- %.3f" % (np.mean(result), np.std(result)))
    print("\n\n",result)
    


if __name__ == "__main__":
    main()
