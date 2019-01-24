#!../venv/bin/python3
import numpy as np
import pandas as pd
import datacontrol
import sklearn
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from sklearn.model_selection import StratifiedKFold


NetworArchitecture = [100, 100, 100, 100]
activation = "relu"
lastlayeractivation = "softmax"
eta = 0.01
n_epoch = 500
momentum = 0.5
nesterov = True
batch_size = 200


def newModel(D_in, D_out):
    return torch.nn.Sequential(
        torch.nn.Linear(D_in, NetworArchitecture[0]),
        torch.nn.ReLU(),
        torch.nn.Linear(NetworArchitecture[0], D_out),
    )

def main():
    TrainingData = datacontrol.readFile("../data/ML-CUP18-TR.csv")
    TestData = datacontrol.readFile("../data/ML-CUP18-TS.csv")
    x_train, y_train = datacontrol.divide(TrainingData)
    # preprocessing
    x_train = sklearn.preprocessing.scale(
        x_train, axis=0, with_mean=True, with_std=True, copy=True)
    input_dimention = x_train.shape[1]
    output_dimention = y_train.shape[1]
    model = newModel(input_dimention, output_dimention)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = torch.nn.MSELoss(reduction='sum')
    for epoch in range(n_epoch):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model(x_train)
        # Compute and print loss.
        loss = loss_fn(y_pred, y_train)
        print(loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()


if __name__ == "__main__":
    main()

main()
