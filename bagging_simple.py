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
    num_classifiers = 1
    train, target = splitDataset(dataset)
    model_bagging = BaggingClassifier(base_estimator=classifier, n_estimators=num_classifiers)
    model_bagging.fit(train, target)
    return model_bagging

def init():

    classifier_model = ClassifierModel("Decision Tree", DecisionTreeClassifier())
    #classifier_model = ClassifierModel("Naive Bayes", GaussianNB())
    #classifier_model = ClassifierModel("SVM", svm.SVC())

    dataset_train, dataset_test = loadDatasets()
    test, target = splitDataset(dataset_test)

    model = buildBagging(dataset_train, classifier_model.classifier)
    model.predict(test)
    print(classifier_model.name, ": ", round(model.score(test, target), 2))

init()
