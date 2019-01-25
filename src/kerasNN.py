#!../venv/bin/python3
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
import keras
import numpy as np
import pandas as pd
import datacontrol
import sklearn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import validation


NetworArchitecture = [100, 30, 30]
activation = "relu"
eta = 0.001
DropOutInput = 0
DropOutHiddenLayer = 0
epochs = 500
momentum = 0.4
nesterov = False
batch_size = 128
Not_yet_printed = True

#######################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# If use drop out we should use the 
#  kernel_initializer='normal', kernel_constraint=maxnorm(3))
# as suggested on dropout paper
######################################################

from functools import wraps
import inspect

def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper

class KerasNN():
    @initializer
    def __init__(self, NetworArchitecture = NetworArchitecture, activation = activation, DropOutHiddenLayer = DropOutHiddenLayer, DropOutInput = DropOutInput,epochs = epochs, eta = eta, momentum = momentum, nesterov = nesterov, batch_size = batch_size):
        pass

    def createModel(self, input_dimention, output_dimention):
        model = Sequential()
        global Not_yet_printed
        if self.DropOutInput> 0:
            if Not_yet_printed:
                print("DropOut in the input layer, value: ", self.DropOutInput)
            model.add(Dropout(self.DropOutInput, input_shape=(input_dimention,)))
            model.add(Dense(kernel_initializer='normal', kernel_constraint=maxnorm(3), units=self.NetworArchitecture[0], activation=activation, input_dim=input_dimention))
        else:
            if Not_yet_printed:
                print("No Drop out in the input layer")
            model.add(Dense(units=self.NetworArchitecture[0], activation=self.activation, input_dim=input_dimention))
        for node in self.NetworArchitecture[1:]:
            if self.DropOutInput > 0 :
                model.add(Dense(kernel_initializer='normal', kernel_constraint=maxnorm(3),
                                units=node, activation=self.activation, input_dim=input_dimention))
            elif self.DropOutHiddenLayer > 0:
                if Not_yet_printed:
                    print("DropOut un the hidden layer. Value: ", self.DropOutHiddenLayer)
                model.add(Dropout(self.DropOutHiddenLayer))
                model.add(Dense(kernel_initializer='normal', kernel_constraint=maxnorm(3),
                                units=node, activation=self.activation, input_dim=input_dimention))
            else:
                if Not_yet_printed:
                    print("No drop out in the hidden layer")
                model.add(Dense(units=node, activation=self.activation))
            Not_yet_printed = False
        model.add(Dense(units=output_dimention, activation = "linear"))
        model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(lr=self.eta, momentum=self.momentum, nesterov=self.nesterov))
        return model

    def showresult(self, history):
        loss_values = history.history['loss']
        val_loss_value = history.history['val_loss']
        epochs_array = range(1, self.epochs + 1)
        plt.plot(epochs_array, loss_values, 'go', label='training loss')
        plt.plot(epochs_array, val_loss_value, 'b', label='validation loss')
        plt.title('yee')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


    def trainValidation(self, X_train, Y_train, x_val, y_val, plot = False):
        start_time = time.time()
        input_dimention = X_train.shape[1]
        output_dimention = Y_train.shape[1]
        model = self.createModel(input_dimention, output_dimention)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        x_val = scaler.transform(x_val)
        history = model.fit(X_train, Y_train, shuffle = True, validation_data=(x_val, y_val), epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
        y_2 = model.predict(x_val)
        scores = validation.MeanEuclidianError(y_2, y_val)
        print("%s: %.2f" % (model.metrics_names[0], scores))
        print("time: %2f"%(time.time()-start_time))
        val_loss_value = history.history['val_loss']
        min_loss_valuation = np.argmin(val_loss_value)
        if plot:
            self.showresult(history)

    def trainCV(self, x_train , y_train, plot = False):
        print("kerasNN\n")
        with tf.device('/device:GPU:0'):
            start_time = time.time()
            #data creation
            # preprocessing
            #x_train = sklearn.preprocessing.scale(x_train, axis=0, with_mean=True, with_std=True, copy=True)
            input_dimention = x_train.shape[1]
            output_dimention = y_train.shape[1]
            # Now we will use k_fold in order to validate the model
            kf = KFold(n_splits=4)
            
            # scaler for NN
            scaler = StandardScaler()
            validationError = []
            trainingError = []
            history_array = []
            for train, test in kf.split(x_train):
                model = self.createModel(input_dimention, output_dimention)
                X_train, x_test, Y_train, y_test = x_train[train], x_train[test], y_train[train], y_train[test]
                # Now we will sclae the data
                # We will fit the scaler with the training set and apply the trasformation also
                # to the test data
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                x_test = scaler.transform(x_test)
                history = model.fit(X_train, Y_train, shuffle = True, validation_data=(x_test, y_test), epochs=self.epochs, batch_size=self.batch_size, verbose=0)
                y_2 = model.predict(x_test)
                scores = validation.MeanEuclidianError(y_2, y_test)
                validationError.append(scores)

                y_2 = model.predict(X_train)
                scores = validation.MeanEuclidianError(y_2, Y_train)
                trainingError.append(scores)
                history_array.append(history)
                #loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
                #print("%s: %.2f" % (model.metrics_names[0], scores))
                #print(scores)
                #val_loss_value = history.history['val_loss']
                #min_loss_valuation = np.argmin(val_loss_value)
                #print("Min loss on validation set was: %.2f on epoch %d" %(val_loss_value[min_loss_valuation], min_loss_valuation))
            print("\n Time: %.2f" % (time.time() - start_time))
            print("%.2f (+/- %.2f)" %(np.mean(validationError), np.std(validationError)))
            print("%.2f (+/- %.2f)" %(np.mean(trainingError), np.std(trainingError)))
            if plot:
                for res in history_array:
                    self.showresult(res)
            return validationError, trainingError


def main():
    TrainingData = datacontrol.readFile("../data/ML-CUP18-TR.csv")
    X, Y = datacontrol.divide(TrainingData)
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=42)
    NN = KerasNN()
    NN.trainValidation(X_train, y_train,X_test, y_test, plot=True)

if __name__ == "__main__":
    main()
