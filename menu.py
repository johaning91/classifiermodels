#menu py
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_auc_score, plot_roc_curve
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import math

weight_model = False
builded = False
optimized =False 
array_clasiffier = []
array_built = []
array_predicted = []
models = []
cross = False

class ClassifierModel:
    def __init__(self, name, classifier, weigth_init = None):
        self.name = name
        self.classifier = classifier
        self.weight_init = weigth_init

def getFileData():
    url = "estudiantes_balanceado_simulado.xlsx"
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
  global train
  num_classifiers = 10
  train, target = splitDataset(dataset)
  model_bagging = BaggingClassifier(base_estimator=classifier, n_estimators=num_classifiers)
  model_bagging.fit(train, target)
  print ('Construyendo Bagging  ', classifier)
  return model_bagging

def buildBaggingCross(dataset, classifier):
  global train
  global target
  target = []
  num_classifiers = 10
  train, target = splitDataset(dataset)
  model_baggingC = BaggingClassifier(base_estimator=classifier, n_estimators=num_classifiers)
  print ('Construyendo Bagging validación cruzada ', classifier)
  return model_baggingC

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

def menu():
  strs = ('Menú para la creación del modelo híbrido\n'
          'Seleccione una opción\n'
          'Presione 1 para agregar Bagging al modelo\n'
          'Presione 2 para utilizar validación cruzada\n'
          'Presione 3 para optimizar el modelo\n'
          'Presione 4 para ver el estado del modelo\n'
          'Presione 0 para salir : \n')
  choice = input(strs)
  return int(choice)


def menuBagging():
  tec  = ('\nSeleccione la tecnica que desea agregar\n'
          'Presione 1 para agregar Bagging de Arboles de decisión\n'
          'Presione 2 para agregar Bagging de SVM\n'
          'Presione 3 para agregar Bagging de Naive Bayes\n'
          'Presione 0 para regresar al menú principal\n')  
  choice = input(tec)
  return int(choice)

def menuOptimization():
  tec  = ('\nSeleccione la tecnica con la que desea optimizar el modelo\n'
            'Presione 1 para optimizar con Hill Climbing \n'
            'Presione 2 para optimizar con Simulated Annealing\n'
            'Presione 3 para optimizar con Step Grid\n'
            'Presione 0 para regresar al menú principal\n')  
  choice = input(tec)
  return int(choice)

def menuOptimizeWeightFalse():
  w_inicial  = ('\nEl modelo esta con pesos por defecto, desea cambiarlos\n'
          'Presione 1 para cambiarlos\n'
          'Presione 2 para continuar con la optimización del modelo \n'
          'Presione 0 para volver al menú anterior\n')
  
  choice = input(w_inicial)
  return int(choice)

def setWeightToModel():
  global weight_model 
  print('Cantidad de modelos internos: '+str(len(array_clasiffier)))

  for item in array_clasiffier:
    w_inicial = ('Recuerde que los pesos de los modelos internos deben sumar 100%\n'
                 'Digite el porcentaje de peso para el modelo: '+item.name+': ')
    choice = input(w_inicial)
    

    item.weight_init = round(int(choice)/100, 2)
  weight_model = True
  print('Pesos establecidos correctamente')
  optimize()

def menuTrian():
  val  = ('\nPresione 1 para utilizar validación cruzada\n'
          'Presione 2 para NO utilizar validación cruzada\n'
          'Presione 0 para regresar al menú principal\n')  
  choice = input(val)
  return int(choice)

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

def optimizeHill(N):
  global array_weight
  global target
  global test
  global dataset_train
  global dataset_test
  global builded
  global optimized
  array_predicted =[]

  dataset = getFileData()
  dataset_train, dataset_test = loadDatasets()
  test, target = splitDataset(dataset_test)
  
  train, target_c = splitDataset(dataset)

  for item in array_clasiffier:
    if cross:
      model = buildBaggingCross(dataset, item.classifier)
      models.append(model)
      model_predicted = cross_val_predict(model, X=train, y=target_c, cv=10)
      array_predicted.append(model_predicted)
      model_score = cross_val_score(model, X=train, y=target_c, cv=10)
      print(item.name,": ", round(model_score.mean(), 2))
    else:
      model = buildBagging(dataset_train, item.classifier)
      models.append(model)
      array_predicted.append(model.predict(test))
      print(item.name,": ", round(model.score(test, target), 2))

  print('Modelo híbrido contruido ...\n')  
  builded = True
  precision_x, y_predicted = modelPrecision(array_predicted, array_weight, target)
  for i in range(N):
    array_s = optimizeWeight(array_weight)
    precision_s, y_predicted = modelPrecision(array_predicted, array_weight, target)

    if precision_s > precision_x:
        precision_x = precision_s
        array_weight = array_s

  print("Precision del modelo híbrido :", precision_x)
  print("###################################")
  print (len(target), len(y_predicted))
  print(confusion_matrix(target, y_predicted))
  print("Area bajo la curva", roc_auc_score(target, y_predicted))
  print("Recuerdi", recall_score(target, y_predicted))
  optimized = True
  return

def simulated(N):
  
  global optimized
  global builded
  dataset = getFileData()
  dataset_train, dataset_test = loadDatasets()
  test, target = splitDataset(dataset_test)

  train, target_c = splitDataset(dataset)

  array_predicted = []
  models = []
  array_score = []
  for item in array_clasiffier:
    if cross:
      model = buildBaggingCross(dataset, item.classifier)
      models.append(model)
      model_predicted = cross_val_predict(model, X=train, y=target_c, cv=10)
      array_predicted.append(model_predicted)
      model_score = cross_val_score(model, X=train, y=target_c, cv=10)
      print(item.name, ": ", round(model_score.mean(), 2))
    else:
      model = buildBagging(dataset_train, item.classifier)
      predicted = model.predict(test)
      array_predicted.append(predicted)
      array_score.append(round(model.score(test, target), 4))
      print(item.name,": ", round(model.score(test, target), 4))

  array_weight = initializeWeight(array_clasiffier, weight_model)  # obtiene los pesos iniciales
  print("Pesos iniciales: ", array_weight)
  if cross:
    current_score, y_predicted = modelPrecision(array_predicted, array_weight, target_c)
  else:
    current_score, y_predicted = modelPrecision(array_predicted, array_weight, target)
  
  solution_weight = array_weight[:]
  print('Modelo híbrido contruido ...\n')  
  builded = True
  print("Precisión inicial:", current_score)
  for i in range(N):
  #while current_score <= final_score:
      getNeighbor(array_weight)
      if cross:
        precision_model, y_predicted = modelPrecision(array_predicted, array_weight, target_c)
      else:
        precision_model, y_predicted = modelPrecision(array_predicted, array_weight, target)
      
      score_diff = precision_model - current_score
      if score_diff > 0:
        solution_weight = array_weight[:]
        if cross:
          precision_model, y_predicted = modelPrecision(array_predicted, array_weight, target_c)
        else:
          precision_model, y_predicted = modelPrecision(array_predicted, array_weight, target)
      else:
          if random.uniform(0,1) < math.exp(-score_diff / current_score):
            solution_weight = array_weight[:]

  print("Pesos finales:", solution_weight)
  print("Precisión final:", current_score)
  print("###################################")
  if cross:
    matrix = confusion_matrix(target_c, y_predicted)
    precision_tp = matrix[0][0] / (matrix[0][0]+matrix[1][0])
    print(matrix)
    print("Precisión verdaderos positivos:", precision_tp)
    print("Area bajo la curva", roc_auc_score(target_c, y_predicted))
    print("Recuerdo", recall_score(target_c, y_predicted))
  else:
    matrix = confusion_matrix(target, y_predicted)
    precision_tp = matrix[0][0] / (matrix[0][0]+matrix[1][0])
    print(matrix)
    print("Precisión verdaderos positivos:", precision_tp)
    print("Area bajo la curva", roc_auc_score(target, y_predicted))
    print("Recuerdo", recall_score(target, y_predicted))
  optimized = True
  return 

def grid(N):

  global array_weight
  global target
  global test
  global dataset_train
  global dataset_test
  global builded
  global optimized
  array_predicted =[]

  dataset = getFileData()
  dataset_train, dataset_test = loadDatasets()
  test, target = splitDataset(dataset_test)
  
  train, target_c = splitDataset(dataset)

  for item in array_clasiffier:
    if cross:
      model = buildBaggingCross(dataset, item.classifier)
      models.append(model)
      model_predicted = cross_val_predict(model, X=train, y=target_c, cv=10)
      array_predicted.append(model_predicted)
      model_score = cross_val_score(model, X=train, y=target_c, cv=10)
      print(item.name,": ", round(model_score.mean(), 2))
    else:
      model = buildBagging(dataset_train, item.classifier)
      models.append(model)
      array_predicted.append(model.predict(test))
      print(item.name,": ", round(model.score(test, target), 2))

  print('Modelo híbrido contruido ...\n')  
  builded = True

  array_weight = initializeWeight(array_clasiffier, weight_model) # obtiene los pesos iniciales
  print("Initial weights: ", array_weight)

  #obtiene la precion del modelo
  if cross:
    precision_x, y_predicted = modelPrecision(array_predicted, array_weight, target_c)
  else:
    precision_x, y_predicted = modelPrecision(array_predicted, array_weight, target)
  
  print("Precision Model inicial:", precision_x)
  best_weight = array_weight[:]
  for i in range(N):
      optimizeWeight(array_weight)
      if cross:
        precision_s, y_predicted = modelPrecision(array_predicted, array_weight, target_c)
      else:
        precision_s, y_predicted = modelPrecision(array_predicted, array_weight, target)

      if precision_s > precision_x:
          precision_x = precision_s
          best_weight = array_weight[:]

  print("Pesos finales:", best_weight)
  print("Precisión final:", precision_x)
  print("###################################")
  print("###################################")
  if cross:
    matrix = confusion_matrix(target_c, y_predicted)
    precision_tp = matrix[0][0] / (matrix[0][0] + matrix[1][0])
    print(matrix)
    print("Precisión verdaderos positivos:", precision_tp)
    print("Area bajo la curva", roc_auc_score(target_c, y_predicted))
    print("Recuerdo", recall_score(target_c, y_predicted))
    #precisionPaso
    precisionGrid(array_predicted, target_c, array_weight)
  else:
    matrix = confusion_matrix(target, y_predicted)
    precision_tp = matrix[0][0] / (matrix[0][0] + matrix[1][0])
    print(matrix)
    print("Precisión verdaderos positivos:", precision_tp)
    print("Area bajo la curva", roc_auc_score(target, y_predicted))
    print("Recuerdo", recall_score(target, y_predicted))
    #precisionPaso
    precisionGrid(array_predicted, target, array_weight)
  optimized = True
  return 

def precisionGrid(array_predicted, target,  array_weight):
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

def optimize():
  iterations = ('\n Digite el número de iteraciones que desea realizar en la optimización :')
  choice = input(iterations)
  N = int(choice)
  weights = initializeWeight(array_clasiffier, weight_model)
  
  while True:
    sel_optimizer = menuOptimization()
    if sel_optimizer == 1:
      print('Pesos iniciales: ', weights)
      print('optimizando...')
      optimizeHill(N)
      break
    elif sel_optimizer == 2:
      simulated(N)
    elif sel_optimizer == 3:
      grid(N)
    elif sel_optimizer == 0:
      break

def setWeight():
  while True:
    choice = menuOptimizeWeightFalse()
    if choice == 1:
      setWeightToModel()
      break
    elif choice == 2:
      optimize()
    elif choice == 0:
      break

def addBagging():
  while True:
    sel = menuBagging()
    if sel == 1:
      addBaggingArboles()
    elif sel == 2:
      addBaggingSVM()
    elif sel == 3:
      addBaggingNBayes()
    elif sel == 0:
      break

def train():
  global cross
  while True:
    sel = menuTrian()
    if sel == 1:
      cross = True
      break
    elif sel == 2:
      cross = False
      break

    elif sel == 0:
      break

def addBaggingArboles():
  array_clasiffier.append(ClassifierModel('Decision Tree', DecisionTreeClassifier(), 0.1))
  print('Bagging de Arboles agregado exitosamente')

def addBaggingSVM():
  array_clasiffier.append(ClassifierModel('SVM', svm.SVC(), 0.1))
  print('Bagging de SVM agregado exitosamente')

def addBaggingNBayes():
  array_clasiffier.append(ClassifierModel('Naive Bayes', GaussianNB(), 0.1))
  print('Bagging de Naive Bayes agregado exitosamente')

def buildBaggingMHSR():
  global test
  global train

  global array_built
  dataset_train, dataset_test = loadDatasets()
  test, target = splitDataset(dataset_test)
  for item in array_clasiffier:
    model = buildBagging(dataset_train, item.classifier)
    models.append(model)
    #predicted = model.predict(test)
    #array_built.append(predicted)
  print('Modelo híbrido contruido ...\n')

  return model


def buildBaggingMHSRCross():
  global test
  global train
  global array_built
  dataset = getFileData()
  train, target = splitDataset(dataset)
  for item in array_clasiffier:
    model = buildBaggingCross(dataset, item.classifier)
    models.append(model)
    #predicted = cross_val_predict(model, X=train, y=target, cv=10)
    #array_built.append(predicted)
  print('Modelo híbrido contruido ...\n')

  return model
  

def initializeWeight(array_classifier, weight_model):
  global array_weight
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

def estadoComponente():
  print('Clasificadores:' )
  if(len(array_clasiffier)==0):
    print('sin clasificadores')
  for clasificador in array_clasiffier:
    info_clasifier = ''
    info_clasifier += 'Clasificador: '+clasificador.name
    if(weight_model==True):
      info_clasifier += ', weigth: '+str(clasificador.weight_init)+' \n'
    print(info_clasifier)
  print('Modelo con pesos por defecto ?:')
  print(not weight_model)
  print ('Esta construido el modelo?')
  print(builded)
  print ('Esta optimizado?')
  print(optimized)
  print('\n')

def init():

  while True:
    choice = menu()
    if choice == 1:
      addBagging()
    elif choice == 2:
      va_cros= train()
    elif choice == 3:

      if not weight_model:
        setWeight()
      else:
        optimize()
        
    elif choice == 4:
      estadoComponente()
    elif choice == 6:
      results()
    elif choice == 0:
        break

init()