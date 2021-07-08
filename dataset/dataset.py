import pandas as pd
import numpy as np
import xlsxwriter

def init():
    periodo = "2018_2019"
    file_estudiantes = pd.read_excel("estudiantes2018_2019.xlsx", "datos")
    data_estudiantes = file_estudiantes.values

    file_desercion = pd.read_excel("desercion.xlsx", "datos")
    data_desercion = file_desercion.values

    ####################
    """admitidos_20161 = pd.read_excel("admitidos20161.xlsx", "Exportar Hoja de Trabajo").values
    admitidos_20162 = pd.read_excel("admitidos20162.xlsx", "Exportar Hoja de Trabajo").values
    admitidos_20171 = pd.read_excel("admitidos20171.xlsx", "Exportar Hoja de Trabajo").values
    admitidos_20172 = pd.read_excel("admitidos20172.xlsx", "Exportar Hoja de Trabajo").values
    admitidos_20181 = pd.read_excel("admitidos20181.xlsx", "Exportar Hoja de Trabajo").values

    admitidos = [*admitidos_20161, *admitidos_20162, *admitidos_20171, *admitidos_20172, *admitidos_20181]"""

    ##############################
    estudiante_final = []
    total_desertores = 0
    for item_estudiante in data_estudiantes:
        desertor = []
        admitido = []
        estudiante_desertor = []
        for item_desercion in data_desercion:
            if item_estudiante[0] == item_desercion[0]:
                desertor = item_desercion[:]
                total_desertores += 1
                break
        estudiante_desertor = item_estudiante[:]
        if len(desertor) > 0:
            estudiante_desertor = [*item_estudiante, *desertor]

        """else:
            desertor = ["","","","","","","","","","","","","","","","","","","","","","","","","","","","","","","",""]
            estudiante_desertor = [*item_estudiante, *desertor]

        for item_admitido in admitidos:
            if item_estudiante[1] == item_admitido[1]:
                admitido = item_admitido[:]
                break
        if len(admitido) > 0:
            estudiante_desertor = [*estudiante_desertor, *admitido]"""

        estudiante_final.append(estudiante_desertor)

    print(total_desertores)
    df = pd.DataFrame(estudiante_final)
    writer = pd.ExcelWriter('dataset_final'+periodo+'.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

    #print(data_estudiantes)
    #print(data_desercion)
    #print(data_estudiantes)

init()