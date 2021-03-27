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
    num_classifiers = 3
    train, target = splitDataset(dataset)
    model_bagging = BaggingClassifier(base_estimator=classifier, n_estimators=num_classifiers)
    model_bagging.fit(train, target)
    return model_bagging

def modelPrecision(array_predicted, array_weight, target):
    umbral = 0.5
    count_hits = 0
    for i in range(len(array_predicted[0])):
        value = 0
        array_result = []
        for j in range(len(array_weight)):
            array_result.append(array_predicted[j][i] * array_weight[j])
        total = sum(array_result)
        if(total > umbral):
            value = 1
        if(value == target[i]):
            count_hits += 1
    return round(count_hits / len(target), 2)

def optimizeWeight(array_x):
    p = random.randint(0, (len(array_x)-1))
    a = random.uniform(-0.1, 0.1)

    array_x[p] = array_x[p] + a
    if array_x[p] < 0:
        array_x[p] = 0
    if array_x[p] > 1:
        array_x[p] = 1

    total = sum(array_x)
    for i in range(len(array_x)):
        array_x[i] = array_x[i] / total
    return array_x

def initializeWeight(array_classifier, weight_model):
    array_weight = []
    if weight_model == False:
        sum_value = 0
        value = round(1/len(array_classifier), 2)
        for i in range(len(array_classifier)):
            if i == (len(array_classifier) -1):
                value = round((1 - sum_value), 2)
            array_weight.append(value)
            sum_value += value
    else:
        for classifier in array_classifier:
            if classifier.weight_init is not None:
                array_weight.append(classifier.weight_init)
    return array_weight


def init():

    N = 1000  # Numero de iteraciones
    precision_umbral = 0.85
    weight_model = True # definimos si pasamos los pesos de los modelos

    array_clasiffier = [ClassifierModel("Decision Tree", DecisionTreeClassifier(), 0.4),
                        ClassifierModel("Naive Bayes", RandomForestClassifier(), 0.3),
                        ClassifierModel("SVM", GaussianNB(), 0.3)]

    dataset_train, dataset_test = loadDatasets()
    test, target = splitDataset(dataset_test)

    array_predicted = []
    for item in array_clasiffier:
        model = buildBagging(dataset_train, item.classifier)
        array_predicted.append(model.predict(test))
        print(item.name,": ", round(model.score(test, target), 2))

    array_weight = initializeWeight(array_clasiffier, weight_model) # obtiene los pesos iniciales
    print("Initial weights: ", array_weight)

    precision_x = modelPrecision(array_predicted, array_weight, target)

    for i in range(N):
        array_s = optimizeWeight(array_weight)
        precision_s = modelPrecision(array_predicted, array_weight, target)

        if precision_s > precision_x:
            precision_x = precision_s
            array_weight = array_s

    print("Precision Model:", precision_x)

init()
