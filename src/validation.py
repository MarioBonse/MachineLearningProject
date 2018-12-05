from sklearn.model_selection import KFold
import numpy as np

# function that train the algorithm with the function passed
#
def kFoldCross(trainCallback, scoreCallback, x_train, y_train, n_splits=4):
    kf = KFold(n_splits=n_splits)
    result = []
    for train, test in kf.split(x_train):
        X_train, x_test, Y_train, y_test = x_train[train], x_train[test], y_train[train], y_train[test]
        trainCallback(X_train, Y_train)
        score = scoreCallback(x_test, y_test)
        result.append(score)
        #print("loss: %.2f"%(score))
    return result
        
