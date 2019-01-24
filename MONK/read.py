import numpy as numpy
import pandas as pd 

# index = 1, 2, 3
# type = "test"|"train"
def readMonk(index, types):
    if index not in [1,2,3]:
        print("INDEX PROBLEM")
        return False
    if (types != "test") and (types != "train"):
        print("type error ")
        return False 
    data = pd.read_csv('monks-'+str(index)+'.'+types+'.txt', sep=" ",header = None)
    data = data.drop(data.columns[0], axis=1)
    data = data.drop(data.columns[7], axis=1)
    df_numpy = data.values[:]
    y = df_numpy[:,0]
    x = df_numpy[:,1:]
    return x,y
    