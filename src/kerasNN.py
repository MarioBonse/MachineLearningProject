#!/usr/bin/python3
from keras.models import Sequential
from keras.layers import Dense
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
import validation


NetworArchitecture = [50]
activation = "relu"
eta = 0.001
epochs = 500
momentum = 0.9
nesterov = True
batch_size = 400

def createModel(input_dimention, output_dimention):
    model = Sequential()
    model.add(Dense(units=NetworArchitecture[0], activation=activation, input_dim=input_dimention))
    for node in NetworArchitecture[1:]:
        model.add(Dense(units=node, activation=activation))
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
    acc_values = history.history['acc']
    val_acc_value = history.history['val_acc']
    plt.plot(epochs_array, acc_values, 'y', label='training accuracy')
    plt.plot(epochs_array, val_acc_value, 'r', label='validation accurcy')
    plt.title('yee')
    plt.xlabel('epochs')
    plt.ylabel('acc')
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
        model = createModel(input_dimention, output_dimention)
        # Now we will use k_fold in order to validate the model
        kf = KFold(n_splits=5)
        result = []
        history_array = []
        for train, test in kf.split(x_train):
            X_train, x_test, Y_train, y_test = x_train[train], x_train[test], y_train[train], y_train[test]
            history = model.fit(X_train, Y_train,epochs=epochs, batch_size=batch_size, verbose=0)
            # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
            scores = model.evaluate(x_test, y_test, verbose=0)
            result.append(scores)
            history_array.append(history)
            #loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
            #showresult(history)
            print("%s: %.2f" % (model.metrics_names[0], scores))
        print("\n Time: %.2f" % (time.time() - start_time))
        print("%.2f (+/- %.2f)" %(np.mean(result), np.std(result)))

if __name__ == "__main__":
    main()
