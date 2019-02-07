import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import trapz
import datetime
import sklearn.preprocessing
from sklearn.utils import shuffle


def createDevAndTest(TrainingData):
    shuffle(TrainingData)
    Dev = TrainingData[:-100]
    Test = TrainingData[-100:]
    pd.DataFrame(Dev).to_csv("../data/Development.csv",header = False)
    pd.DataFrame(Test).to_csv("../data/MyTest.csv",header = False)
    return True

def readFile(name, shuff= False):
    df = pd.read_csv(name, delimiter=',', comment='#', header=None)
    if shuff:
        df = shuffle(df)
        df = df.reset_index(drop=True)
    df = df.astype(float)
    #in order to return numpy object 
    df_numpy = df.values[:,1:]
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
    df = pd.DataFrame(result)
    now = datetime.datetime.now()
    f = open(name, 'w')
    f.write('# Alfredo Bochicchio, Mario Bonsembiante\n')
    f.write('# Kape Vector Machine (KVM)\n')
    f.write('# ML-CUP18\n')
    f.write('# '+str(now.day)+'/'+str(now.month)+'/'+str(now.year)+'\n')
    df.index += 1 
    df.to_csv(f, sep=',', encoding='utf-8', header = False)
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
