import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import trapz



def main():
    name = "../data/ML-CUP18-TR.csv"
    df = pd.read_csv(name, delimiter=',', comment='#', header=None)
    df = df.astype(float)
