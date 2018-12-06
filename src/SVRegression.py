#!/usr/bin/python3
import numpy as np
import pandas as pd
import datacontrol
import sklearn
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import time
import validation

############################################
# SVM doesn't support multiple output
# so I had to use the "multiple output library" of scikitlearn
# link: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html
##################################################

#h Hiper parameters
kernel = 'linear'
C = 1e3
gamma = 0.1
degree = 0 #only if poly kernel
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
    svr = svm.SVR(kernel='rbf', gamma=0.1, coef0=-1, degree=5)
    SVRegressor = MultiOutputRegressor(svr, n_jobs= 2)
    # I can evaluate the model also with cross Validation
    # CrossValidationScores = cross_val_score(SVRegressor, x_train, y_train, cv=5)
    # I can evaluate the model with kfold validation
    scores = validation.kFoldCross(SVRegressor.fit, SVRegressor.predict, x_train, y_train, n_splits=5 )
    scores = np.array(scores)
    print("\n")
    print("Time: %0.2f" % (time.time() - start_time),"\n\nRESULT")
    for i in scores:
        print("loss: %.2f" % (i))
    print("%0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
    #print("Errors are with CrossValidation are: ", CrossValidationScores, "")
    #print("With mean %0.2f and variance %0.2f" %(CrossValidationScores.mean(), CrossValidationScores.std()*2))
    print("HyperParameters: ....")


if __name__ == "__main__":
    main()
