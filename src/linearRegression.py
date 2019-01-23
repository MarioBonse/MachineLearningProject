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
        # scaler
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        x_test = scaler.transform(x_test)

        reg = LinearRegression().fit(X_train, Y_train)
        # result
        y_2 = []
        for i in range(np.shape(x_test)[0]):
            yout = reg.predict(np.asmatrix(x_test[i]))
            y_2.append(yout[0])
            #score += resultmoment
        scoreValidation = validation.MeanEuclidianError(np.array(y_2), y_test)
        resultVal.append(scoreValidation)

        y_2 = []
        score = 0
        for i in range(np.shape(X_train)[0]):
            yout = reg.predict(np.asmatrix(X_train[i]))[0]
            y_2.append(yout)
            #resultmoment = (yout - Y_train[i, 0])**2
            #score += resultmoment
        scoreTest = validation.MeanEuclidianError(Y_train, np.array(y_2))
        resultTest.append(scoreTest)
        print(time.time() - startTime)
        #loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
    print("Mean validation: %.2f +- %.3f" %(np.mean(resultVal), np.std(resultVal)))
    print("Mean Test: %.2f +- %.3f" %(np.mean(resultTest), np.std(resultTest)))

if __name__ == "__main__":
    main()
