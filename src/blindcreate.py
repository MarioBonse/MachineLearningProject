#!../venv/bin/python3
import pandas as pd
import numpy as np
import seaborn as sns
import kerasNN
import datacontrol
import validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import time

def main():
    TrainingData = datacontrol.readFile("../data/ML-CUP18-TR.csv")
    BlindSet = datacontrol.readFile("../data/ML-CUP18-TS.csv")
    X, Y = datacontrol.divide(TrainingData)
    scaler = StandardScaler()
    start = time.time()
    NN = kerasNN.KerasNN(NetworArchitecture = [625, 625, 625, 625, 625], activation = "relu", eta = 0.00015, momentum = 0.99, epochs = 5000, DropOutHiddenLayer = 0.2)
    model = NN.createModel()
    scaler.fit(X)
    X = scaler.transform(X)
    BlindSet = scaler.transform(BlindSet)
    model.fit(X, Y, shuffle = True, epochs=NN.epochs, batch_size=NN.batch_size)
    YT = model.predict(BlindSet)
    print(YT.shape)
    datacontrol.writeOutput(YT, "../data/FinalResultBlindTest.csv")

if __name__ == "__main__":
    main()
