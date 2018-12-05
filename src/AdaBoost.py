#!/usr/bin/python3
import sklearn
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
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
    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300)
    SVRegressor = MultiOutputRegressor(regr_2, n_jobs=2)
    scores = validation.kFoldCross(SVRegressor.fit, SVRegressor.score, x_train, y_train)
    print("Time: %0.2f" % (time.time() - start_time))
    print("Errors are: ", scores, "")
    scores = np.array(scores)
    print("With mean %0.2f and variance %0.2f" %
          (scores.mean(), scores.std()*2))
    print("Hiper_parameters: ....")


if __name__ == "__main__":
    main()
