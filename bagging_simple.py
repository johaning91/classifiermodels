from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pandas as pd
from numpy import random
import numpy as np
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

def getFileData(filename):
    data_sheet = pd.read_excel(filename)
    dataset = data_sheet.values
    return dataset

def loadDatasets():
    dataset = getFileData("estudiantes_balanceado.xlsx")
    train, test = train_test_split(dataset, test_size=0.30, random_state=1)
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

def init():

    classifier_model = ClassifierModel("Árbol de decisión", DecisionTreeClassifier())
    #classifier_model = ClassifierModel("Naive Bayes", GaussianNB())
    #classifier_model = ClassifierModel("Support Vector Machine", svm.SVC())

    dataset_train, dataset_test = loadDatasets()
    test, target = splitDataset(dataset_test)

    model = buildBagging(dataset_train, classifier_model.classifier)
    predicted = model.predict(test)

    plot_confusion_matrix(model, test, target)
    plot_roc_curve(model, test, target)
    matrix = confusion_matrix(target, predicted)
    precision_tp = matrix[0][0] / (matrix[0][0] + matrix[1][0])

    print("Modelo Bagging Simple")
    print("Clasificador: ", classifier_model.name)
    print("Precisión:", round(model.score(test, target), 2))
    print("Precision TP:", precision_tp)
    print("Recall", recall_score(target, predicted))
    print("AUC", roc_auc_score(target, predicted))
    print(matrix)
    #plt.show()

init()
