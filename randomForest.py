from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm
import pandas as pd
from numpy import random
import numpy as np

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
    train, test = train_test_split(dataset, test_size=0.30, random_state=0)
    return train, test

def splitDataset(dataset):
    size = len(dataset[0]) - 1
    data_train = dataset[:, 0: size]
    data_target = dataset[:, size]
    return data_train, data_target

def buildBagging(dataset, classifier):
    num_classifiers = 1
    train, target = splitDataset(dataset)
    model_bagging = BaggingClassifier(base_estimator=classifier, n_estimators=num_classifiers)
    model_bagging.fit(train, target)
    return model_bagging

def init():

    """dataset_train, dataset_test = loadDatasets()
    train, train_target = splitDataset(dataset_train)
    test, test_target = splitDataset(dataset_test)

    model = RandomForestClassifier(max_depth=50, n_estimators=100, criterion="entropy", max_features="sqrt")
    model.fit(train, train_target)
    model.predict(test)"""

    dataset = getFileData()
    train, test = splitDataset(dataset)

    cross = cross_val_score(RandomForestClassifier(max_depth=200, max_features="sqrt"), X=train, y=test, cv=10)
    print(cross.mean())
    ###print("Random Forest: ", round(model.score(test, test_target), 4))

init()
