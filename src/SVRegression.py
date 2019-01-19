#!/usr/bin/python3
import HyperParameterTester
import datacontrol

# mail file for the gridsearch
def main():
    #data creation
    try:
        TrainingData = datacontrol.readFile("../data/ML-CUP18-TR.csv")
    except:
        TrainingData = datacontrol.readFile("data/ML-CUP18-TR.csv")
    x_train, y_train = datacontrol.divide(TrainingData)
    HP = HyperParameterTester.HyperParameterTesterSVM(5)
    HP.simulate(x_train, y_train)
    HP.sort()
    HP.saveCSV()
    


if __name__ == "__main__":
    main()
