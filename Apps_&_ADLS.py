##############ADLS INTERACTION#################

#Add the following lines to import the required modules#############################

## Use this for Azure AD authentication
from msrestazure.azure_active_directory import AADTokenCredentials

## Required for Azure Data Lake Store account management
from azure.mgmt.datalake.store import DataLakeStoreAccountManagementClient
from azure.mgmt.datalake.store.models import DataLakeStoreAccount

## Required for Azure Data Lake Store filesystem management
from azure.datalake.store import core, lib, multithread

# Common Azure imports
import adal
from azure.mgmt.resource.resources import ResourceManagementClient
from azure.mgmt.resource.resources.models import ResourceGroup

## Use these as needed for your application
import logging, pprint, uuid, time

#############End-user authentication with multi-factor authentication
#############For account management

# authority_host_url = "https://login.microsoftonline.com"
# tenant = "FILL-IN-HERE"
# authority_url = authority_host_url + '/' + tenant
# client_id = 'FILL-IN-HERE'
# redirect = 'urn:ietf:wg:oauth:2.0:oob'
# RESOURCE = 'https://management.core.windows.net/'
#
# context = adal.AuthenticationContext(authority_url)
# code = context.acquire_user_code(RESOURCE, client_id)
# print(code['message'])
# mgmt_token = context.acquire_token_with_device_code(RESOURCE, code, client_id)
# armCreds = AADTokenCredentials(mgmt_token, client_id, resource = RESOURCE)

##############################
#For ADLS End-user authentication
# adlCreds = lib.auth(tenant_id='01af65c0-d4ba-43ec-a72d-6375a87c4c3e', username = 'rosingh@flowserve.com', password= 'password', resource= "adl://serveflowdata.azuredatalakestore.net")
                    # resource = 'https://microsoft.com/devicelogin')


#Service-to-service authentication with client secret for filesystem operations
adlCreds = lib.auth(tenant_id = '01af65c0-d4ba-43ec-a72d-6375a87c4c3e',
                    client_secret = '4QFXIvzbO69/5Dx7S4+8r4p44OurYIIy6ZmTzpty0Qs=',
                    client_id = '45ac04b0-dfd7-4d6f-ad12-3adc1dd8e450',
                    resource = 'https://datalake.azure.net/')

#Access ADLS
adl = core.AzureDLFileSystem(store_name = 'serveflowdata', token = adlCreds)

# adl.mkdir('/Kalamazoo_Data/')
from nptdms import TdmsFile
import pandas as pd

f = adl.open('Dayton_Data/full_trim/C_High/C1FN8_accelerometer.tdms')
f_csv = adl.open('Dayton_Data/MetadataofPump2017-12-21T00-04-36.csv')
data = TdmsFile(file= f).as_dataframe()
data = pd.read_csv(f_csv)



##################FOLDER_PIRATE########################

import shutil, os, glob, datetime
now = datetime.datetime.now()
source = r'C:\Users\10154861\Desktop\Test'
destination = r'C:\Users\10154861\Desktop\Test\End Folder/' + str(now.year)+'_'+'Month'+str(now.month)+'_Day'+str(now.day)

if not os.path.exists(destination):
    os.makedirs(destination)

files = glob.glob(source + "\*.txt")            #Choose file-type

for f in files:
    file = os.path.basename(f)
    print("Stealing: " +file)
    shutil.move(f, destination)


####################READ MULTIPLEFILES FROM FOLDERS##############

import glob, os

path = r"C:\Users\10154861\Desktop"
fileNames = glob.glob(path + "\*.txt")

for f in fileNames:
    file = os.path.basename(f)
    print("Reading: " +file)


######################READ TDMS FILES##################

from nptdms import TdmsFile

f = r'C:\Users\10154861\Desktop\file.tdms'
df = TdmsFile(f).as_dataframe()


###################CONNECT TO POSTGRES##############
import psycopg2 as pg
import pandas.io.sql as psql


try:
    connection = pg.connect("dbname='R&D Data' user='postgres' host='localhost' password='postgres'")
    print("DB Connection Success")
except:
    print ("Unable to connect to DB")

df = psql.read_sql('SELECT * FROM "DaytonData_SeriesA_5pT" WHERE pump_state = 1 fetch first 10000 rows only', connection)

#####################FFT#################3333333333
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nptdms import TdmsFile
#1800rpm

f = r"C:\Users\10154861\Desktop\Work\1. Analytics\Rohit's Private Repo\A1TN1_accelerometer.tdms"

data = TdmsFile(file= f).as_dataframe()
list(data)

data.rename(columns = {"/'Untitled'/'Outboard Horizontal'" :"Horizontal", "/'Untitled'/'Outboard Vertical'":"Vertical", "/'Untitled'/'Axial'" : "Axial"}, inplace = True)
# plt.plot(data['Horizontal'][1:25600])

FFT_Sample_Rate = 2560
sampleRate = 25600          # Data collected every second

#FFT Sample
time_series = data['Horizontal'][0:25600]
# time_series.head()

sampleRate = 25600      #Data collected every second
f = 1/sampleRate

#Y-axis
fft = np.fft.fft(time_series)
fftscaled = abs(fft/len(time_series))

#X-axis
fftfreq = abs(np.fft.fftfreq(len(fftscaled), f))

plt.plot(fftfreq, fftscaled)
# plt.ylim(0,0.006)
# plt.xlim(40,70)
plt.show()

