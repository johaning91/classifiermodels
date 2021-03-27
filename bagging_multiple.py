from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pandas as pd
from numpy import random
import numpy as np
from enum import Enum

class abandonmentType:
    NO = 0
    YES = 1

class ClassifierModel:
    def __init__(self, name, classifier, weigth_init = None):
        self.name = name
        self.classifier = classifier
        self.weight_init = weigth_init

def getFileData():
    url = "chrun.xlsx"
    data_sheet = pd.read_excel(url)
    dataset = data_sheet.values
    return dataset

def loadDatasets():
    dataset = getFileData()
    train, test = train_test_split(dataset, test_size=0.30)
    return train, test

def splitDataset(dataset):
    size = len(dataset[0]) - 1
    data_train = dataset[:, 0: size]
    data_target = dataset[:, size]
    return data_train, data_target

def buildBagging(dataset, classifier):
    num_classifiers = 10
    train, target = splitDataset(dataset)
    model_bagging = BaggingClassifier(base_estimator=classifier, n_estimators=num_classifiers)
    model_bagging.fit(train, target)
    return model_bagging

def modelPrecisionByVote(array_predicted, target):
    count_hits = 0
    num_predicted = len(array_predicted)
    for i in range(len(array_predicted[0])):
        abandonment = 0
        value = abandonmentType.NO
        for j in range(num_predicted):
            if array_predicted[j][i] == abandonmentType.YES:
                abandonment += 1
        if abandonment > (num_predicted - abandonment):
            value = abandonmentType.YES
        if value == target[i]:
            count_hits += 1
    return round(count_hits / len(target), 2)

def init():
    array_clasiffier = [ClassifierModel("Decision Tree", DecisionTreeClassifier(), 0.55),
                        ClassifierModel("Decision Tree 2", DecisionTreeClassifier(), 0.20),
                        ClassifierModel("Decision Tree 3", DecisionTreeClassifier(), 0.20),
                        ClassifierModel("Decision Tree 4", DecisionTreeClassifier(), 0.20)]

    dataset_train, dataset_test = loadDatasets()
    test, target = splitDataset(dataset_test)

    array_predicted = []
    for item in array_clasiffier:
        model = buildBagging(dataset_train, item.classifier)
        array_predicted.append(model.predict(test))
        print(item.name,": ", round(model.score(test, target), 2))

    precision = modelPrecisionByVote(array_predicted, target)
    print("Precision Model:", precision)

init()
