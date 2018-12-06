#!/usr/bin/python3
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
import validation


NetworArchitecture = [100, 100, 100]
activation = "relu"
eta = 0.001
DropOutInput = 0
DropOutHiddenLayer = 0.2
epochs = 4000
momentum = 0.9
nesterov = True
batch_size = 256
Not_yet_printed = True

#######################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# If use drop out we should use the 
#  kernel_initializer='normal', kernel_constraint=maxnorm(3))
# as suggested on dropout paper
######################################################
def createModel(input_dimention, output_dimention):
    model = Sequential()
    global Not_yet_printed
    if DropOutInput> 0:
        if Not_yet_printed:
            print("DropOut in the input layer, value: ", DropOutInput)
        model.add(Dropout(DropOutInput, input_shape=(input_dimention,)))
        model.add(Dense(kernel_initializer='normal', kernel_constraint=maxnorm(3), units=NetworArchitecture[0], activation=activation, input_dim=input_dimention))
    else:
        if Not_yet_printed:
            print("No Drop out in the input layer")
        model.add(Dense(units=NetworArchitecture[0], activation=activation, input_dim=input_dimention))
    for node in NetworArchitecture[1:]:
        if DropOutInput > 0 :
            model.add(Dense(kernel_initializer='normal', kernel_constraint=maxnorm(3),
                            units=node, activation=activation, input_dim=input_dimention))
        elif DropOutHiddenLayer > 0:
            if Not_yet_printed:
                print("DropOut un the hidden layer. Value: ", DropOutHiddenLayer)
            model.add(Dropout(DropOutHiddenLayer))
            model.add(Dense(kernel_initializer='normal', kernel_constraint=maxnorm(3),
                            units=node, activation=activation, input_dim=input_dimention))
        else:
            if Not_yet_printed:
                print("No drop out in the hidden layer")
            model.add(Dense(units=node, activation=activation))
        Not_yet_printed = False
    model.add(Dense(units=output_dimention))
    model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(lr=eta, momentum=momentum, nesterov=nesterov))
    return model

def showresult(history):
    loss_values = history.history['loss']
    val_loss_value = history.history['val_loss']
    epochs_array = range(1, epochs + 1)
    plt.plot(epochs_array, loss_values, 'go', label='training loss')
    plt.plot(epochs_array, val_loss_value, 'b', label='validation loss')
    plt.title('yee')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def main():
    print("kerasNN\n")
    with tf.device('/device:GPU:0'):
        start_time = time.time()
        #data creation
        TrainingData = datacontrol.readFile("../data/ML-CUP18-TR.csv")
        x_train, y_train = datacontrol.divide(TrainingData)
        # preprocessing
        #x_train = sklearn.preprocessing.scale(x_train, axis=0, with_mean=True, with_std=True, copy=True)
        input_dimention = x_train.shape[1]
        output_dimention = y_train.shape[1]
        # Now we will use k_fold in order to validate the model
        kf = KFold(n_splits=5)
        # scaler for NN
        scaler = StandardScaler()
        result = []
        history_array = []
        for train, test in kf.split(x_train):
            model = createModel(input_dimention, output_dimention)
            X_train, x_test, Y_train, y_test = x_train[train], x_train[test], y_train[train], y_train[test]
            # Now we will sclae the data
            # We will fit the scaler with the training set and apply the trasformation also
            # to the test data
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            x_test = scaler.transform(x_test)
            history = model.fit(X_train, Y_train, validation_data= (x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=0)
            # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
            scores = model.evaluate(x_test, y_test, verbose=0)
            result.append(scores)
            history_array.append(history)
            #loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
            print("%s: %.2f" % (model.metrics_names[0], scores))
            print(scores)
            val_loss_value = history.history['val_loss']
            min_loss_valuation = np.argmin(val_loss_value)
            print("Min loss on validation set was: %.2f on epoch %d" %(val_loss_value[min_loss_valuation], min_loss_valuation))

        #for history in history_array:
        #    showresult(history)
        print("\n Time: %.2f" % (time.time() - start_time))
        print("%.2f (+/- %.2f)" %(np.mean(result), np.std(result)))

if __name__ == "__main__":
    main()
