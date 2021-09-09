from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pandas as pd
from numpy import random
import numpy as np
from enum import Enum
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score, plot_roc_curve
import matplotlib.pyplot as plt

class abandonmentType:
    NO = 0
    YES = 1

class ClassifierModel:
    def __init__(self, name, classifier, weigth_init = None):
        self.name = name
        self.classifier = classifier
        self.weight_init = weigth_init

def getFileData():
    url = "estudiantes_balanceado.xlsx"
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
    y_predicted = []
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
        y_predicted.append(value)
    return round(count_hits / len(target), 2), y_predicted

def init():
    print("Modelo Bagging Híbrido MHSR")
    array_clasiffier = [ClassifierModel("Árbol de decisión", DecisionTreeClassifier(max_depth=1), 0.55),
                        ClassifierModel("Naive Bayes", GaussianNB(), 0.20),
                        ClassifierModel("Support Vector Machine", KNeighborsClassifier(), 0.20)]

    dataset_train, dataset_test = loadDatasets()
    test, target = splitDataset(dataset_test)

    array_predicted = []
    models=[]
    for item in array_clasiffier:
        model = buildBagging(dataset_train, item.classifier)
        models.append(model)
        array_predicted.append(model.predict(test))
        #print(item.name,": ", round(model.score(test, target), 2))

    precision, y_predicted = modelPrecisionByVote(array_predicted, target)
    #print("Precision Model:", precision)

    plot_confusion_matrix(models[0], test, target)
    plot_roc_curve(models[0], test, target)
    #plt.show()

    matrix = confusion_matrix(target, y_predicted)
    precision_tp = matrix[0][0] / (matrix[0][0] + matrix[1][0])
    #print(matrix)
    print("Precision TP:", precision_tp)
    print("AUC", roc_auc_score(target, y_predicted))
    print("Recall", recall_score(target, y_predicted))

init()
