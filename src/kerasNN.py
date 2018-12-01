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

NetworArchitecture = [100, 1000]
activation = "relu"
lastlayeractivation = "softmax"
eta = 0.1
epochs = 100
momentum = 0.1
nesterov = False
batch_size = 32

def main():
    print("kerasNN\n")

    with tf.device('/device:GPU:2'):
        #data creation
        TrainingData = datacontrol.readFile("../data/ML-CUP18-TR.csv")
        TestData = datacontrol.readFile("../data/ML-CUP18-TS.csv")
        x_train, y_train = datacontrol.divide(TrainingData)
        input_dimention = x_train.shape[1]
        output_dimention = y_train.shape[1]
        # preprocessing
        x_train = sklearn.preprocessing.scale(
            x_train, axis=0, with_mean=True, with_std=True, copy=True)
        #now define the model
        model = Sequential()
        model.add(Dense(units=NetworArchitecture[0], activation=activation, input_dim=input_dimention))
        for node in NetworArchitecture[1:]:
            model.add(Dense(units=node, activation=activation))
        model.add(Dense(activation=activation, output_dim=output_dimention))
        model.compile(metrics=['acc'], loss=keras.losses.mean_absolute_error,
                    optimizer=keras.optimizers.SGD(lr=eta, momentum=momentum, nesterov=nesterov))
        # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
        history = model.fit(x_train, y_train, validation_split=0.2,epochs=epochs, batch_size=batch_size, verbose=0)
        loss_and_metrics = model.evaluate(x_train, y_train, batch_size=128)
        loss_values = history.history['loss']
        val_loss_value = history.history['val_loss']
        epochs_array = range(1, epochs + 1)
        plt.plot(epochs_array, loss_values, 'bo', label = 'training loss')
        plt.plot(epochs_array, val_loss_value, 'b', label = 'validation loss')
        plt.title('yee')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


if __name__ == "__main__":
    main()
