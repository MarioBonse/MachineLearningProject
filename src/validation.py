from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np

# function that train the algorithm with the function passed
#
def kFoldCross(trainCallback, predictcallback, x_train, y_train, n_splits=4, preprocessing = True):
    kf = KFold(n_splits=n_splits)
    if preprocessing:
        scaler = StandardScaler()
    result = []
    for train, test in kf.split(x_train):
        X_train, x_test, Y_train, y_test = x_train[train], x_train[test], y_train[train], y_train[test]
        if preprocessing:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            x_test = scaler.transform(x_test)
        trainCallback(X_train, Y_train)
        y_predicted = predictcallback(x_test)
        loss = (((y_predicted - y_test)**2).sum())/y_test.size
        result.append(loss)
    return result
        
