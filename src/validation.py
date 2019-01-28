from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import RepeatedKFold

# function that train the algorithm with the function passed
#
def kFoldCross(trainCallback, predictcallback, x_train, y_train, n_splits=4, preprocessing = True):
    kf = KFold(n_splits=n_splits)
    if preprocessing:
        scaler = StandardScaler()
    valLoss = []
    TRLoss = []

    for train, test in kf.split(x_train):
        X_train, x_test, Y_train, y_test = x_train[train], x_train[test], y_train[train], y_train[test]
        if preprocessing:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            x_test = scaler.transform(x_test)
        trainCallback(X_train, Y_train)
        y_predicted = predictcallback(x_test)
        loss_val = MeanEuclidianError(y_predicted, y_test)
        loss_training = MeanEuclidianError(predictcallback(X_train), Y_train)
        valLoss.append(loss_val)
        TRLoss.append( loss_training)
    return valLoss, TRLoss


# RepeatedKFold
# 

def RepeatedKFoldCross(trainCallback, predictcallback, x_train, y_train, n_splits=5, preprocessing=True, n_repeats=4):
    kf = RepeatedKFold(n_repeats=n_repeats, n_splits=n_splits)
    if preprocessing:
        scaler = StandardScaler()
    valLoss = []
    TRLoss = []
    for train, test in kf.split(x_train):
        X_train, x_test, Y_train, y_test = x_train[train], x_train[test], y_train[train], y_train[test]
        if preprocessing:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            x_test = scaler.transform(x_test)
        trainCallback(X_train, Y_train)
        y_predicted = predictcallback(x_test)
        #mean euclidian error
        loss_val = MeanEuclidianError(y_predicted, y_test)
        loss_training = MeanEuclidianError(predictcallback(X_train), Y_train)
        valLoss.append(loss_val)
        TRLoss.append(loss_training)
        print(loss_val)
    return valLoss, TRLoss

        
def MeanEuclidianError(X,Y):
    #x, y in R2
    #MEE = 1/N(sum(sqrt(x[i,0]-y[i,0]**2 + x[i,1]-y[i,1]**2))
    #can be weitten in a faster way as 1/N(sumnp.sqrt(float(x.dot(x)) - 2 * float(x.dot(y)) + float(y.dot(y))))
    out = 0    
    for x, y in zip(X, Y):
        out +=  np.sqrt(np.linalg.norm(x-y))
    return out/(X.shape[0])
