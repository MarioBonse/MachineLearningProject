#!/usr/bin/python3
import HyperParameterTester
import datacontrol
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import validation
import numpy as np
import matplotlib.pyplot as plt
# plt: ":", "--", "-.", 
# it test the variance if we change the training /valiation division
def main():
    #data creation
    try:
        TrainingData = datacontrol.readFile("../data/Development.csv")
    except:
        TrainingData = datacontrol.readFile("data/Development.csv")
    x_train, y_train = datacontrol.divide(TrainingData)
    svr = svm.SVR(kernel="rbf", gamma="scale", C=56)
    SVRegressor = MultiOutputRegressor(svr, n_jobs=2)
    # I can evaluate the model also with cross Validation
    # CrossValidationScores = cross_val_score(SVRegressor, x_train, y_train, cv=5)
    # I can evaluate the model with kfold validation
    valScore, TrainingScore = validation.RepeatedKFoldCross(
        SVRegressor.fit, SVRegressor.predict,  x_train, y_train, n_repeats=1)
    valScore = np.array(valScore)
    TrainingScore = np.array(TrainingScore)
    print(valScore)
    print("Test = %.2f +/- %.2f" % (np.mean(TrainingScore), np.std(TrainingScore)))
    print("validation = %.2f +/- %.2f" % (np.mean(valScore), np.std(valScore)))
    test_dim = range(0, np.size(valScore))
    plt.plot(test_dim, TrainingScore, 'g', label='Training Score')
    plt.plot(test_dim, valScore, '--b', label='Validation Score')
    plt.title('Training and validation result with RepeatedKFoldCross cross validation')
    plt.xlabel('CrossValidationIndex')
    plt.ylabel('Score')
    plt.legend(['train', 'Validation'], loc='upper left')
    plt.show()

if __name__ == "__main__":
   main() 
