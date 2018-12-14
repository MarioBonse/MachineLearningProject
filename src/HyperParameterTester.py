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
kernels = ['rbf']
C_range = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]
degree_range = [7] #only in poly
coef_range = [1]  # only in ply and sigmoid!
gamma_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
epsilon = 0.001

class HyperParameterSVM:
    def __init__(self, C, gamma, epsilon, degree, kernel, coef):
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.kernel = kernel
        self.coef = coef
    
    def SaveResult(self, valScore, TrainingScore, timeSimulation):
        self.time = timeSimulation
        self.kfold = np.size(valScore)
        self.valResult = valScore
        self.Vmean = np.mean(valScore)
        self.Vdeviation = np.std(valScore)
        self.TRResult = TrainingScore
        self.TRmean = np.mean(TrainingScore)
        self.TRdeviation = np.std(TrainingScore)

    def getValue(self):
        return [self.Vmean, self.Vdeviation, self.TRmean, self.TRdeviation, self.C, self. epsilon, self.gamma, self.degree, self.kernel, self.coef,
                self.time, *self.valResult, *self.TRResult]



class HyperParameterTesterSVM:
    def __init__(self, kfoldDim):
        e = epsilon
        self.kfoldDim = kfoldDim
        self.title = ["Validation mean", "Validation deviation", "Training mean", "Training deviation", "C", "epsilon", "gamma", "degree", "kernel", "coef",
                 "time"]
        for i in range(kfoldDim):
            self.title.append("Validation foldN _"+str(i))
        for i in range(kfoldDim):
            self.title.append("Training foldN _"+str(i))
        gamma = 'scale'
        self.HyperParameterArray = []
        for k in kernels:
            for C in C_range:
                for gamma in gamma_range:
                    if k == 'poly':
                        for d in degree_range:
                            for coeff in coef_range:
                                self.HyperParameterArray.append(HyperParameterSVM(C, gamma, e, d, k, coeff))
                    elif k == 'sigmoid':
                        for coeff in coef_range:
                            self.HyperParameterArray.append(HyperParameterSVM(C, gamma, e, 0, k, coeff))
                    elif k == 'rbf':
                        self.HyperParameterArray.append(HyperParameterSVM(C, gamma, e, 0, k, 0))

    def simulate(self, x_train, y_train):
        writer = csv.writer(open("CSVResult/CSVResultDuringSimulation.csv", 'w'))
        writer.writerow(self.title)
        for simulation in self.HyperParameterArray:
            start_time = time.time()
            svr = svm.SVR(kernel=simulation.kernel, gamma=simulation.gamma, coef0=simulation.coef, degree=simulation.degree, C = simulation.C, epsilon=simulation.epsilon)
            SVRegressor = MultiOutputRegressor(svr, n_jobs=2)
            # I can evaluate the model also with cross Validation
            # CrossValidationScores = cross_val_score(SVRegressor, x_train, y_train, cv=5)
            # I can evaluate the model with kfold validation
            valScore, TrainingScore = validation.kFoldCross(
                SVRegressor.fit, SVRegressor.predict,  x_train, y_train, n_splits=self.kfoldDim)
            valScore = np.array(valScore)
            TrainingScore = np.array(TrainingScore)
            timeSimulation = abs(time.time()-start_time)
            simulation.SaveResult(valScore, TrainingScore, timeSimulation)
            print("\n")
            print("Validation error: %0.2f (+/- %0.2f)" % (valScore.mean(), valScore.std()*2))
            print("Training Error: %0.2f (+/- %0.2f)" %(TrainingScore.mean(), TrainingScore.std()*2))
            print("time = %0.2f" % timeSimulation)
            param = simulation.getValue()
            writer.writerow(param)

    def sort(self):
        self.HyperParameterArray.sort(key=lambda x: x.Vmean)

        
    def saveCSV(self):
        writer = csv.writer(open("CSVResult/CSVResultFinal.csv", 'w'))
        writer.writerow(self.title)
        writer.writerow(["Validation mean", "Validation deviation", "Training mean", "Training deviation", "C", "epsilon", "gamma", "degree", "kernel", "coef",
                         "time"])
        for results in self.HyperParameterArray:
            writer.writerow(results.getValue())

