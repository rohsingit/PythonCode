import glob
import xlrd
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

files = glob.glob(r"C:\Users\10154861\Desktop\Work\1. Python\A. Data\Taneytown\2015\*.*")
df = {'Filename': [], 'Pump_Model': [], 'Serial_Number': [], 'Motor_Number': [], 'Stages': [], 'Discharge_Diameter': [],
      'Impeller_Diameter': [], 'Service_Flow': [], 'Service_Head': [], 'Service_Speed': [], 'Service_SG': [],
      'Flow1': [], 'Flow2': [], 'Flow3': [], 'Flow4': [], 'Flow5': [], 'Flow6': [], 'Flow7': [], 'Flow8': [],
      'Power1': [], 'Power2': [], 'Power3': [], 'Power4': [], 'Power5': [], 'Power6': [], 'Power7': [], 'Power8': [],
      'Discharge_P2': [], 'Discharge_P2': [], 'Discharge_P3': [], 'Discharge_P4': [], 'Discharge_P5': [], 'Discharge_P6': [], 'Discharge_P7': [], 'Discharge_P8': []}
table = pd.DataFrame(df)
#
for filepath in files:
    xl_workbook = xlrd.open_workbook(filepath)
    xl_sheet_Test_Setup = xl_workbook.sheet_by_name('Test Lab Setup')
    xl_sheet_PP = xl_workbook.sheet_by_name('PP')
    # print(xl_sheet_Test_Setup.name)

    model_num = xl_sheet_Test_Setup.cell(9, 0)     #Cell A10
    serial_num = xl_sheet_Test_Setup.cell(10,0)    #Cell A11
    motor_num = xl_sheet_Test_Setup.cell(13,0)     #Cell A14
    stages_num = xl_sheet_Test_Setup.cell(16,0)    #Cell A17
    diacharge_dia= xl_sheet_Test_Setup.cell(18,0)  #Cell A19

    impeller_dia = xl_sheet_PP.cell(57,7)           #Cell H58
    service_flow = xl_sheet_PP.cell(55,14)          #Cell O56
    service_head = xl_sheet_PP.cell(56,14)          #Cell O57
    service_speed = xl_sheet_PP.cell(58, 14)        #Cell O59
    service_sg = xl_sheet_PP.cell(59,14)            #Cell O60

    #Take performance data
    flow1 = xl_sheet_Test_Setup.cell(62, 2)
    flow2 = xl_sheet_Test_Setup.cell(63, 2)
    flow3 = xl_sheet_Test_Setup.cell(64, 2)
    flow4 = xl_sheet_Test_Setup.cell(65, 2)
    flow5 = xl_sheet_Test_Setup.cell(66, 2)
    flow6 = xl_sheet_Test_Setup.cell(67, 2)
    flow7 = xl_sheet_Test_Setup.cell(68, 2)
    flow8 = xl_sheet_Test_Setup.cell(69, 2)

    power1 = xl_sheet_Test_Setup.cell(62, 3)
    power2 = xl_sheet_Test_Setup.cell(63, 3)
    power3 = xl_sheet_Test_Setup.cell(64, 3)
    power4 = xl_sheet_Test_Setup.cell(65, 3)
    power5 = xl_sheet_Test_Setup.cell(66, 3)
    power6 = xl_sheet_Test_Setup.cell(67, 3)
    power7 = xl_sheet_Test_Setup.cell(68, 3)
    power8 = xl_sheet_Test_Setup.cell(69, 3)

    disc_pres1 = xl_sheet_Test_Setup.cell(62, 4)
    disc_pres2 = xl_sheet_Test_Setup.cell(63, 4)
    disc_pres3 = xl_sheet_Test_Setup.cell(64, 4)
    disc_pres4 = xl_sheet_Test_Setup.cell(65, 4)
    disc_pres5 = xl_sheet_Test_Setup.cell(66, 4)
    disc_pres6 = xl_sheet_Test_Setup.cell(67, 4)
    disc_pres7 = xl_sheet_Test_Setup.cell(68, 4)
    disc_pres8 = xl_sheet_Test_Setup.cell(69, 4)

    # print(cell_obj)
    basename = os.path.basename(filepath)
    print(basename)
    table = table.append({'Filename' : basename, 'Pump_Model' : model_num.value, 'Serial_Number' : serial_num.value,
                          'Motor_Number': motor_num.value, 'Stages': stages_num.value, 'Discharge_Diameter': diacharge_dia.value,
                          'Impeller_Diameter': impeller_dia.value, 'Service_Flow': service_flow.value, 'Service_Head': service_head.value,
                          'Service_Speed': service_speed.value, 'Service_SG': service_sg.value,
                          'Flow1': flow1.value, 'Flow2': flow2.value, 'Flow3': flow3.value, 'Flow4': flow4.value,'Flow5': flow5.value,
                          'Flow6': flow6.value,'Flow7': flow7.value,'Flow8': flow8.value,
                          'Power1': power1.value, 'Power2': power2.value, 'Power3': power3.value, 'Power4': power4.value, 'Power5': power5.value,
                          'Power6': power6.value, 'Power7': power7.value, 'Power8': power8.value,
                          'Discharge_P2': disc_pres1.value, 'Discharge_P2': disc_pres2.value, 'Discharge_P3': disc_pres3.value, 'Discharge_P4': disc_pres4.value,
                          'Discharge_P5': disc_pres5.value, 'Discharge_P6': disc_pres6.value, 'Discharge_P7': disc_pres7.value, 'Discharge_P8': disc_pres8.value},
                          ignore_index= True)

print(table.head())
# table.to_csv("TaneyTown_Models_Summary2015.csv")

# table['Service_Head'] = table['Service_Head'].astype(float)
# table['Pump_Model'].value_counts().plot(kind = 'bar', legend = True, title = 'TaneyTown2015 Pump Models')
# table['Pump_Model'].value_counts().head(25).plot(kind = 'bar', title = 'TaneyTown2015 Pump Models')

# pd.plotting.scatter_matrix(table)