import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import trapz
import datetime
import sklearn.preprocessing
from random import shuffle

def readFile(name):
    df = pd.read_csv(name, delimiter=',', comment='#', header=None)
    df = df.astype(float)
    #in order to return numpy object 
    df_numpy = df.values[:, 1:]
    shuffle(df_numpy)
    return df_numpy

# give a dataframe it divide it in two parts
def divide(Data, size = 2):
    X = Data[:, :-size]
    Y = Data[:, -size:]
    return X, Y

def writeOutput(result, name):
    # Name1  Surname1, Name2 Surname2
    # Group Nickname
    # ML-CUP18
    # 02/11/2018
    now = datetime.datetime.now()
    f = open(name, 'a')
    f.write('# Sophie ?, Mario Bonsembiante\n')
    f.write('# Booooo\n')
    f.write('# ML-CUP18\n')
    f.write('# ',str(now.day),'/',str(now.month),'/',str(now.year),'\n')
    result.to_csv(f, sep='\t', encoding='utf-8')
    f.close()

# given a starting point a and ending point b (usually [-1,1]) rescaling the data froin the [a, b] interval
# with the formula 
# x_new = ((x_old - data[0])*(b - a) + a*(data[-1] - data[0]))/(data[-1] - data[0])
def rescaling(data, a = -1, b = 1):
    #data = sort(data)
    d_new = b - a
    start = data[0]
    end = data[-1]
    d_old = end - start
    data = ((data - start)*(d_new) + a *
            (d_old))/(d_old)

def main():
    df = readFile('../data/ML-CUP18-TR.csv')
    df = sklearn.preprocessing.scale(
        df, axis=0, with_mean=True, with_std=True, copy=True)
    #distribution plot
    plt.figure()
    for i in range(1, df.shape[1]):
        sns.distplot(df[i], hist=False, rug=True)
    plt.figure()
    for i in range(df.shape[1] - 2, df.shape[1]):
        sns.distplot(df[i], hist=False, rug=True)
    '''
    for i in range(1, df.shape[1] - 2):
        plt.figure()
        sns.boxplot(df[i])
    for i in range(df.shape[1] - 2, df.shape[1]):
        plt.figure()
        sns.boxplot(df[i])
    '''
    plt.show()

