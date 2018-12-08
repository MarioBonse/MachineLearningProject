#!/usr/bin/python3
import HyperParameterTester
import datacontrol


def main():
    #data creation
    TrainingData = datacontrol.readFile("../data/ML-CUP18-TR.csv")
    x_train, y_train = datacontrol.divide(TrainingData)
    HP = HyperParameterTester.HyperParameterTesterSVM()
    HP.simulate(x_train, y_train)
    HP.sort()
    HP.saveCSV()
    


if __name__ == "__main__":
    main()
