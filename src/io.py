import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import trapz

def readFile(name):
    df = pd.read_csv(name, delimiter=',', comment='#', header=None)
    df = df.astype(float)
    return df

df = readFile('../data/ML-CUP18-TR.csv')

#distribution plot
plt.figure()
for i in range(1, df.shape[1] -2):
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
