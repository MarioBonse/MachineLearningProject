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
    def __init__(self, NetworArchitecture =  [1000], activation =  "relu", alpha = 0.1, Lastactivation = "", n_split = 5):
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

    def train(self):
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
        kf = KFold(n_splits=self.n_split)
        # scaler for NN
        scaler = StandardScaler()
        resultVal = []
        resultTraining = []
        for train, test in kf.split(x_train):
            startTime = time.time()
            model = self.newModel(input_dimention)
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
            reg =  linear_model.Ridge(alpha=self.alpha).fit(NN_output, Y_train)
            y_2 = []
            for i in range(np.shape(x_test)[0]):
                out = model(torch.from_numpy(x_test[i]).float())
                out = out.detach().numpy()
                yout = reg.predict(np.asmatrix(out))
                y_2.append(yout[0])
                #score += resultmoment
            scoreValidation = validation.MeanEuclidianError(np.array(y_2), y_test)
            resultVal.append(scoreValidation)
            y_2 = []
            score = 0
            for i in range(np.shape(X_train)[0]):
                out = model(torch.from_numpy(X_train[i]).float())
                out = out.detach().numpy()
                yout = reg.predict(np.asmatrix(out))
                y_2.append(yout[0])
                #resultmoment = (yout - Y_train[i, 0])**2
                #score += resultmoment
            scoreTest = validation.MeanEuclidianError(Y_train, np.array(y_2))
            resultTraining.append(scoreTest)
            print(time.time() - startTime)
            #loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
        print("Mean validation: %.2f +- %.3f" %(np.mean(resultVal), np.std(resultVal)))
        print("Mean training: %.2f +- %.3f" %(np.mean(resultTraining), np.std(resultTraining)))
        return resultVal, resultTraining
    
def main():
    ellm = elm()
    ellm.train()

if __name__ == "__main__":
    main()
