#!/usr/bin/python3
from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np
import pandas as pd
import datacontrol
import sklearn
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.model_selection import StratifiedKFold


NetworArchitecture = [20, 20]
activation = "relu"
eta = 0.001
epochs = 2000
momentum = 0.9
nesterov = True
batch_size = 400

def createModel(input_dimention, output_dimention):
    model = Sequential()
    model.add(Dense(units=NetworArchitecture[0], activation=activation, input_dim=input_dimention))
    for node in NetworArchitecture[1:]:
        model.add(Dense(units=node, activation=activation))
    model.add(Dense(units=output_dimention))
    model.compile(metrics=['acc'], loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.SGD(lr=eta, momentum=momentum, nesterov=nesterov))
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
        TestData = datacontrol.readFile("../data/ML-CUP18-TS.csv")
        x_train, y_train = datacontrol.divide(TrainingData)
        # preprocessing
        #x_train = sklearn.preprocessing.scale(x_train, axis=0, with_mean=True, with_std=True, copy=True)
        input_dimention = x_train.shape[1]
        output_dimention = y_train.shape[1]
        model = createModel(input_dimention, output_dimention)
        
        # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
        history = model.fit(x_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)
        print(model.summary())
        print("\n Time: %.2f" % (time.time() - start_time))
        #loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
        showresult(history)
        


if __name__ == "__main__":
    main()
