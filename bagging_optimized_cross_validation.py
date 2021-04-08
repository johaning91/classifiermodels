from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
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
    #result = cross_val_predict(model_bagging, X=train, y=target)
    #model_bagging.fit(train, target)
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

def precisionPaso(array_predicted, target):
    array_weight = [0, 0, 0]
    size_weight = len(array_weight)
    file_data = ""
    for pos in (range(size_weight)):
        file_data += "##### Peso "+str(pos+1)+" #####\n"
        for i in range(100):
            item_value = round(((i+1) * 0.01),2)
            weight = round(1 - item_value, 2)
            value = round(weight / (size_weight - 1), 2)
            sum_value = 0
            for j in (range(size_weight)):
                if j == pos:
                    array_weight[j] = item_value
                else:
                    if j == size_weight-1:
                        value = round(weight - sum_value, 2)
                    array_weight[j] = value
                    sum_value += value
            precision = modelPrecision(array_predicted, array_weight, target)
            str_weight = ' '.join(str(e) for e in array_weight)
            file_data += ""+ str_weight + " Precision: "+str(precision)+"\n"

    writeToFile(file_data)

def writeToFile(data):
    f = open("precision.txt", "w+")
    f.write(data)
    f.close()

def init():

    N = 1000  # Numero de iteraciones
    precision_umbral = 0.85
    weight_model = True # definimos si pasamos los pesos de los modelos

    array_clasiffier = [ClassifierModel("Decision Tree", DecisionTreeClassifier(random_state=1), 0.6),
                        ClassifierModel("Naive Bayes", GaussianNB(), 0.2),
                        ClassifierModel("SVM", svm.SVC(), 0.2)]

    dataset = getFileData()
    train, target = splitDataset(dataset)

    array_predicted = []
    for item in array_clasiffier:
        model = buildBagging(dataset, item.classifier)
        array_predicted.append(cross_val_predict(model, X=train, y=target, cv=10))
        model_score = cross_val_score(model, X=train, y=target, cv=10)
        print(item.name,": ", round(model_score.mean(), 2))

    array_weight = initializeWeight(array_clasiffier, weight_model) # obtiene los pesos iniciales
    print("Initial weights: ", array_weight)

    precision_x = modelPrecision(array_predicted, array_weight, target)
    print("Precision Model inicial:", precision_x)
    best_weight = array_weight[:]
    """for i in range(N):
        optimizeWeight(array_weight)
        precision_s = modelPrecision(array_predicted, array_weight, target)

        if precision_s > precision_x:
            precision_x = precision_s
            best_weight = array_weight[:]

    print("Pesos:", best_weight)
    print("Precision Model:", precision_x)
    print("Precision Model2:", modelPrecision(array_predicted, best_weight, target))"""

    #precisionPaso
    precisionPaso(array_predicted, target)

init()
