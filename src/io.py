import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import trapz

def readFile(name):
    with open('../data/ML-CUP18-TR.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        start = False
        df = []
        for row in spamreader:
            if start:
                df.append(row)
            if row == []:
                start = True
    df = pd.DataFrame(df)
    df = df.astype(float)
    return df

df = readFile('../data/ML-CUP18-TR.csv')
#I know it's soo ugly but I haven't found ho to avoid # lines with pandas

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
plt.show()





