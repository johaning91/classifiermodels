from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import pandas as pd
from numpy import random
import numpy as np
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score, plot_roc_curve
import matplotlib.pyplot as plt

class ClassifierModel:
    def __init__(self, name, classifier, weigth_init = None):
        self.name = name
        self.classifier = classifier
        self.weight_init = weigth_init

def getFileData():
    url = "estudiantes2020.xlsx"
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

def modelPrecision(array_predicted, array_weight, target):
    umbral = 0.5
    count_hits = 0
    y_predicted = []
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
        y_predicted.append(value)
    return round(count_hits / len(target), 4), y_predicted

def getNeighbor(array_x):

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
    weight_model = False # definimos si pasamos los pesos de los modelos

    array_clasiffier = [ClassifierModel("Decision Tree", DecisionTreeClassifier(random_state=1, max_depth=20), 0.4),
                        ClassifierModel("Naive Bayes", GaussianNB(var_smoothing=5), 0.3),
                        ClassifierModel("SVM", KNeighborsClassifier(), 0.3)]

    dataset = getFileData()
    train, target = splitDataset(dataset)

    array_predicted = []
    models = []
    for item in array_clasiffier:
        model = buildBagging(dataset, item.classifier)
        models.append(model)
        model_predicted = cross_val_predict(model, X=train, y=target, cv=10)
        array_predicted.append(model_predicted)
        model_score = cross_val_score(model, X=train, y=target, cv=10)
        print(item.name, ": ", round(model_score.mean(), 2))

    array_weight = initializeWeight(array_clasiffier, weight_model)  # obtiene los pesos iniciales
    print("Initial weights: ", array_weight)
    current_score, y_predicted = modelPrecision(array_predicted, array_weight, target)
    solution_weight = array_weight[:]

    print("Initial precision:", current_score)
    for i in range(N):
    #while current_score <= final_score:
        getNeighbor(array_weight)
        precision_model, y_predicted = modelPrecision(array_predicted, array_weight, target)
        score_diff = precision_model - current_score
        if score_diff > 0:
            solution_weight = array_weight[:]
            current_score, y_predicted = modelPrecision(array_predicted, array_weight, target)
        else:
            if random.uniform(0,1) < math.exp(-score_diff / current_score):
                solution_weight = array_weight[:]

    print("Final weights:", solution_weight)
    print("Final precision:", current_score)
    print("###################################")
    matrix = confusion_matrix(target, y_predicted)
    precision_tp = matrix[0][0] / (matrix[0][0]+matrix[1][0])
    print(matrix)
    print("Precision TP:", precision_tp)
    print("AUC", roc_auc_score(target, y_predicted))
    print("Recall", recall_score(target, y_predicted))

init()
