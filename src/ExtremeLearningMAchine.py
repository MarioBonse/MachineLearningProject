#!../venv/bin/python3
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
from sklearn import linear_model
import torch

NetworArchitecture = [1000, 10000]
activation = "relu" #are : "relu", "sigmoid", "tanh"
Lastactivation = ""
alpha = 0.1

class elm():
    def __init__(self, NetworArchitecture =  [2000], activation =  "relu", alpha = 0.1, Lastactivation = "", n_split = 5):
        self.NetworArchitecture = NetworArchitecture
        self.activation = activation #are : "relu", "sigmoid", "tanh"
        self.Lastactivation = Lastactivation
        self.alpha = alpha
        self.n_split = n_split

    def newModel(self, D_in):
        network = []
        size = len(self.NetworArchitecture)
        network.append(torch.nn.Linear(D_in, self.NetworArchitecture[0]))
        if self.activation == "sigmoid":
            network.append(torch.nn.Sigmoid())
        elif self.activation == "relu":
            network.append(torch.nn.ReLU())
        elif self.activation == "tanh":
            network.append(torch.nn.Tanh())
        for i in range(1, size-1):
            network.append(torch.nn.Linear(
                self.NetworArchitecture[i-1], self.NetworArchitecture[i]))
            if self.activation == "sigmoid":
                network.append(torch.nn.Sigmoid())
            elif self.activation == "relu":
                network.append(torch.nn.ReLU())
            elif self.activation == "tanh":
                network.append(torch.nn.Tanh())
        network.append(torch.nn.Linear(self.NetworArchitecture[size-2], self.NetworArchitecture[size-1]))
        if self.Lastactivation == "sigmoid":
            network.append(torch.nn.Sigmoid())
        elif self.Lastactivation == "relu":
            network.append(torch.nn.ReLU())
        elif self.Lastactivation == "tanh":
            network.append(torch.nn.Tanh())
        return torch.nn.Sequential(*network)

    def fit(self,X_train, Y_train, scaler = True):
        input_dimention = X_train.shape[1]
        self.model = self.newModel(input_dimention)
        if scaler:
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train)
        else:
            self.scaler = False
        NN_output = []
        for x in range(np.shape(X_train)[0]):
            self.out = self.model(torch.from_numpy(X_train[x]).float())
            self.out = self.out.detach().numpy()
            NN_output.append(self.out)
        NN_output = np.array(NN_output)
        self.reg =  linear_model.Ridge(alpha=self.alpha).fit(NN_output, Y_train)

    def predict(self, x_test):
        if self.scaler:
            x_test = self.scaler.transform(x_test)
        y_out = []
        for i in range(np.shape(x_test)[0]):
            self.out = self.model(torch.from_numpy(x_test[i]).float())
            self.out = self.out.detach().numpy()
            yout = self.reg.predict(np.asmatrix(self.out))
            y_out.append(np.squeeze(yout))
        return np.array(y_out)


    def trainCV(self, x_train, y_train):
        # preprocessing
        #x_train = sklearn.preprocessing.scale(x_train, axis=0, with_mean=True, with_std=True, copy=True)
        input_dimention = x_train.shape[1]
        output_dimention = y_train.shape[1]
        # Now we will use k_fold in order to validate the model
        kf = KFold(n_splits=self.n_split)
        # scaler for NN
        scaler = StandardScaler()
        resultVal = []
        resultTraining = []
        for train, test in kf.split(x_train):
            startTime = time.time()
            X_train, x_test, Y_train, y_test = x_train[train], x_train[test], y_train[train], y_train[test]
            # Now we will sclae the data
            # We will fit the scaler with the training set and apply the trasformation also
            # to the test data
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            x_test = scaler.transform(x_test)
            self.fit(X_train, Y_train)
            y_2 = self.predict(x_test)
            scoreValidation = validation.MeanEuclidianError(y_2, y_test)
            resultVal.append(scoreValidation)
            y_2 = self.predict(X_train)
            scoreTest = validation.MeanEuclidianError(Y_train, y_2)
            resultTraining.append(scoreTest)
        # print the results
        print("Mean validation: %.2f +- %.3f" %(np.mean(resultVal), np.std(resultVal)))
        print("Mean training: %.2f +- %.3f" %(np.mean(resultTraining), np.std(resultTraining)))
        print("Time: %.2f" % (time.time()-startTime))
        return resultVal, resultTraining
    
def main():
    try:
        TrainingData = datacontrol.readFile("data/Development.csv")
    except:
        TrainingData = datacontrol.readFile("../data/Development.csv")
    x_train, y_train = datacontrol.divide(TrainingData)
    ellm = elm()
    ellm.trainCV(x_train, y_train)

if __name__ == "__main__":
    main()
