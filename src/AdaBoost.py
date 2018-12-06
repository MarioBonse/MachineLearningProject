#!/usr/bin/python3
import sklearn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import svm
import matplotlib.pyplot as plt
import time
import datacontrol
import validation
import numpy as np


def main():
    #data creation
    TrainingData = datacontrol.readFile("../data/ML-CUP18-TR.csv")
    #will introduce cross validation then
    x_train, y_train = datacontrol.divide(TrainingData)
    #x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.4, random_state=0)
    # preprocessing
    #x_train = sklearn.preprocessing.scale(x_train, axis=0, with_mean=True, with_std=True, copy=True)
    #########
    start_time = time.time()
    #regr_2 = AdaBoostRegressor(svm.SVR(kernel='rbf', gamma=0.1, coef0=-1, degree=5), n_estimators=100)
    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=1000)
    SVRegressor = MultiOutputRegressor(regr_2, n_jobs=2)
    scores = validation.kFoldCross(SVRegressor.fit, SVRegressor.predict, x_train, y_train)
    print("Time: %0.2f" % (time.time() - start_time), "\n\nRESULT")
    scores = np.array(scores)
    for i in scores:
        print("loss: %.2f" % (i))
    print("%0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
    print("HyperParameters: ....")


if __name__ == "__main__":
    main()
