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
import csv 

###########################################
# SVM doesn't support multiple output
# so I had to use the "multiple output library" of scikitlearn
# link: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html
##################################################

# Hiper parameters
kernels = ['rbf']#, 'poly', 'sigmoid']
C_range = [0.1, 0.1, 1, 10, 100]
degree_range = [1]#, 2, 3, 4] #only in poly
coef_range = [0]#, 1, 2, 10]  # only in ply and sigmoid!
gamma_range = 'scale'
epsilon = [0.1, 0.01, 0.001, 0.5]

class HyperParameterSVM:
    def __init__(self, C, gamma, epsilon, degree, kernel, coef):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.kernel = kernel
        self.coef = coef
    
    def SaveResult(self, result, time):
        self.time = time
        self.kfold = np.size(result)
        self.result = result
        self.mean = np.mean(result)
        self.deviation = np.std(result)

    def getValue(self):
        return [self.mean, self.deviation, self.C, self. epsilon, self.gamma, self.degree, self.kernel, self.coef,
          self.time, self.kfold]



class HyperParameterTesterSVM:
    def __init__(self):
        gamma = 'scale'
        self.HyperParameterArray = []
        for k in kernels:
            for C in C_range:
                for e in epsilon:
                    if k == 'poly':
                        for d in degree_range:
                            for coeff in coef_range:
                                self.HyperParameterArray.append(HyperParameterSVM(C, gamma, e, d, k, coeff))
                    elif k == 'sigmoid':
                        for coeff in coef_range:
                            self.HyperParameterArray.append(HyperParameterSVM(C, gamma, e, 0, k, coeff))
                    else:
                        self.HyperParameterArray.append(HyperParameterSVM(C, gamma, e, 0, k, 0))

    def simulate(self, x_train, y_train):
        for simulation in self.HyperParameterArray:
            start_time = time.time()
            svr = svm.SVR(kernel=simulation.kernel, gamma=simulation.gamma, coef0=simulation.coef, degree=simulation.degree, C = simulation.C, epsilon=simulation.epsilon)
            SVRegressor = MultiOutputRegressor(svr, n_jobs=2)
            # I can evaluate the model also with cross Validation
            # CrossValidationScores = cross_val_score(SVRegressor, x_train, y_train, cv=5)
            # I can evaluate the model with kfold validation
            scores = validation.kFoldCross(
            SVRegressor.fit, SVRegressor.predict,  x_train, y_train, n_splits=5)
            scores = np.array(scores)
            timeSimulation = abs(time.time()-start_time)
            simulation.SaveResult(scores, timeSimulation)
            print("\n")
            print("%0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
            param = SVRegressor.get_params()
            print(param, "time: %0.2f" % timeSimulation)

    def sort(self):
        self.HyperParameterArray.sort(key=lambda x: x.mean)

        
    def saveCSV(self):
        writer = csv.writer(open("CSVResult/CSVResult.csv", 'w'))
        writer.writerow(["mean", "deviation", "C", "epsilon", "gamma", "degree", "kernel", "coef",
                         "time", "kfold"])
        for results in self.HyperParameterArray:
            writer.writerow(results.getValue())

