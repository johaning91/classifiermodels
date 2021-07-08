import pandas as pd
import numpy as np
from numpy import random
import xlsxwriter

def removeDuplicateData():
    filename = 'dataset_inicial'
    file_excel = pd.read_excel(filename + ".xlsx")
    data = file_excel.values
    new_data = []
    for item in data:
        count = 0
        for item2 in data:
           if item[0] == item2[0]:
               count +=1
        if count <= 1:
            new_data.append(item)

    df = pd.DataFrame(new_data)
    writer = pd.ExcelWriter(filename + '_out.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


def simulateData():
    filename = 'DATASET_SIMULADO'
    file_excel = pd.read_excel(filename + ".xlsx")
    data = file_excel.values

    for item in data:
        value = round(random.uniform(2.6, 3.5),1)
        if item[14] == 'NO':
            value = round(random.uniform(3.3, 4),1)
        item[12] = value

    n_work = round(9779 * 0.24) #24% estudia y trabaja en colombia
    item = 0
    size_data = len(data)
    while(item < n_work):
        position = random.randint(0, (size_data - 1))
        if data[position][13] == 0:
            data[position][13] = 1
            item+=1

    df = pd.DataFrame(data)
    writer = pd.ExcelWriter(filename+'_out.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()


def convertDataToInt():
    num_data = 5102
    file_excel = pd.read_excel("dataset.xlsx")
    data = file_excel.values
    array_new = []
    array_size = len(data)

    """for item in data:

    df = pd.DataFrame(array_new)
    writer = pd.ExcelWriter('dataset_filtro.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()"""

def cnn():
    num_data = 2630
    file_name = "dataset_inicial"
    file_excel = pd.read_excel(file_name+".xlsx")
    data = file_excel.values
    array_new = []
    array_size = len(data)
    col_index = len(data[0])-1
    index_id = 0
    classA = 'NO'
    classB = 'SI'
    while len(array_new) < num_data:
        position = random.randint(0, (array_size - 1))
        if data[position][col_index] == classA:
            exist = False
            for item in array_new:
                if item[index_id] == data[position][index_id]:
                    exist = True
                    break
            if exist == False:
                array_new.append(data[position])

    for item in data:
        if item[col_index] == classB:
            position = random.randint(0, (len(array_new) - 1))
            array_new.insert(position, item)

    df = pd.DataFrame(array_new)
    writer = pd.ExcelWriter(file_name+'_balanceado.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

def init():
    cnn()
    #convertDataToInt()
    #simulateData()
    #removeDuplicateData()

init()