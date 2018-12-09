#!/usr/bin/python3
import math
import itertools
import optunity
import optunity.metrics
import sklearn.svm
import datacontrol
from sklearn.multioutput import MultiOutputRegressor

space = {'kernel': {'linear': {'C': [0, 100]},
                    'rbf': {'gamma': [0, 50], 'C': [1, 100]},
                    'poly': {'degree': [2, 5], 'C': [1000, 20000], 'coef0': [0, 1]}
                    }
        }

TrainingData = datacontrol.readFile("../data/ML-CUP18-TR.csv")
x_train, y_train = datacontrol.divide(TrainingData)

outer_cv = optunity.cross_validated(x=x_train, y=y_train, num_folds=3)


def compute_mse_standard(x_train, y_train, x_test, y_test):
    """Computes MSE of an SVR with RBF kernel and default hyperparameters.
    """
    svmodel = sklearn.svm.SVR()
    SVRegressor = MultiOutputRegressor(svmodel, n_jobs=2)
    model = SVRegressor.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return optunity.metrics.mse(y_test, predictions)


# wrap with outer cross-validation
compute_mse_standard = outer_cv(compute_mse_standard)


def compute_mse_all_tuned(x_train, y_train, x_test, y_test):
    """Computes MSE of an SVR with RBF kernel and optimized hyperparameters."""

    # define objective function for tuning
    @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
    def tune_cv(x_train, y_train, x_test, y_test, kernel, C, gamma, degree, coef0):
        if kernel == 'linear':
            model = sklearn.svm.SVR(kernel=kernel, C=C)
            SVRegressor = MultiOutputRegressor(model, n_jobs=2)
        elif kernel == 'poly':
            model = sklearn.svm.SVR(
                kernel=kernel, C=C, degree=degree, coef0=coef0)
            SVRegressor = MultiOutputRegressor(model, n_jobs=2)
        elif kernel == 'rbf':
            model = sklearn.svm.SVR(kernel=kernel, C=C, gamma=gamma)
            SVRegressor = MultiOutputRegressor(model, n_jobs=2)
        SVRegressor.fit(x_train, y_train)

        predictions = SVRegressor.predict(x_test)
        return (optunity.metrics.mse(y_test[:, 0], predictions[:, 0])+
                    optunity.metrics.mse(y_test[:, 1], predictions[:, 1]))

    # optimize parameters
    optimal_pars, _, _ = optunity.minimize_structured(tune_cv, num_evals=150, search_space=space)

    # remove hyperparameters with None value from optimal pars
    for k, v in optimal_pars.items():
        if v is None:
            del optimal_pars[k]
    print("optimal hyperparameters: " + str(optimal_pars))
    tuned_model = sklearn.svm.SVR(**optimal_pars).fit(x_train, y_train)
    SVRegressor = MultiOutputRegressor(tuned_model, n_jobs=2)
    model = SVRegressor.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return optunity.metrics.mse(y_test, predictions)


# wrap with outer cross-validation
compute_mse_all_tuned = outer_cv(compute_mse_all_tuned)
compute_mse_all_tuned()
