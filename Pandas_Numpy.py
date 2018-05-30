import pandas as pd
import matplotlib.pyplot as plt

##READ FILES/GENERATE DATA
#From CSV or TXT
path = r"filename.csv, sep = ",", header = None"
path = r"filename.txt, sep = "|""   #Header will be loaded
df = pd.read_csv(path, encoding='cp1252', sep = ';')    #Encoding if default 'utf-8' not working
df = pd.read_csv(path)
df.isnull().sum()

##CREATE LIST FROM EXCEL
path = r'C:\Users\rohitsingh\filename.xlsx'
List = pd.read_excel(path)
List = list(List['Column_Name'])      #Creates a list with members of column name

##SELECT SUBSET FROM DATAFRAME
Subset_DataFrame1 = df.loc[df['Column_Name'] == 'Value']

some_list = ('Value1', 'Value2', 'Value3')        #Create a list
Subset_DataFrame2 = df.loc[df['Column_Name'].isin(some_list)

##EXPORT TO CSV
df.to_csv("name_of_file.csv")

#LOAD AS TIME SERIES
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv("path", parse_dates= ['columnName'], index_col= 'columnName', date_parser= dateparse) #or index_col = [0]
csv = pd.read_csv(filepath, encoding= 'cp1252', parse_dates= [0], index_col= [0], sep = ';', date_parser= dateparse)

##DATE AND INDEX MANIPULATION
##Convert time from UNIX to usual format
df['Time'] = pd.to_datetime(df['Timestamp'], unit = 's')
df.set_index('Time', inplace= True)
del df['Timestamp']

#CREATE DATAFRAME
# df = {'Filename': [], 'Pump_Model': [], 'Serial_Number': [], 'Motor_Number': []}
# table = pd.DataFrame(df)

##CHECK FOR MISSING CELLS/FILL FORWARD
# df.isnull().sum()
# df = df.fillna(method= 'ffill')


##Preview data
# print(df.head())
# print(df.tail())
# print(df.describe())
# print(df['CF_13'].describe())
# print(df.dtypes)
# print(df.index)
                           
##CHECK VALUE_COUNTS
df['Column_Name'].value_counts().plot(kind = 'bar')   #Or assign to variable --> value_count = ...

# len(array_name)
##SELECT DATA COLUMNS ETC
df.iloc[0:4]        #Show rows 0 to 3
df.iloc[0:4,1]      #Show rows 0 to 3 from column 1
df.iloc[:,0]        #Show all rows column 0
df.iloc[1,]         #Show first row all columns

##MANIPULATE COLUMNS
# list(df)            #List column names
# df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})
# col_my_order = ['C','B','A']
# df = df[col_my_order]       #Rearrange columns in order

#CONCAT, SET INDEX
# arima_df = pd.concat((timestamp, cf_100), axis = 1)
# arima_df.set_index('Timestamp', inplace = True)

#CHECK ARRAYS
# np.array_equal(array1 , array2) #True or False

##PLOT USING PANDAS
# df.plot()                 #Plot everything
# df.plot(y = '3852')       #Plot a line chart of a column using pandas' plot method
# df.plot.box(rot =  90)

##PLOT USING MATPLOTLIB
# plt.plot(df['3852'])
# plt.plot(df['3852'][0:10000])
# plt.axhline(0.55, color = 'red')      #Put after plotting to avoid x-axis rescaling
# plt.title("CF_32 plot")
# plt.xticks(rotation=90)
# plt.show()
# plt.plot(df)

###SUBPLOTTING
# fig = plt.figure(1)
# fig.subplots_adjust(hspace=0.4, wspace=0.4)
#
# for i in range(0, len(col_my_order1)):
#     print("plotting:", col_my_order1[i])
#     fig.add_subplot(2, 4, i+1)
#     plt.title(col_my_order1[i])
#     plt.xticks(rotation=90)
#     plt.plot(df[col_my_order1[i]])
#     plt.axhline(y=col_my_order1_limits[i], color='red')
# plt.show()


#####PLOT ROLLING OR RESAMPLE MEAN##############

# arima_df.rolling('3600s).mean().plot(title = "Hourly Avg")     #Hourly avg
# arima_df.rolling('86400s').mean().plot(title = "Daily Avg")    #Daily avg
# arima_df.rolling('604800s').mean().plot(title = "Weekly Avg")         #Weekly
# arima_df.rolling('2592000s').mean().plot(title = "30-day Avg")   #30-day avg
# arima_data = arima_df.rolling(window='2592000s').mean()

# arima_data_resample = arima_df.resample('2592000s').mean()        #30-day resample -downsample
# arima_data_resample = arima_df.resample('D').mean()                 #1day -downsample
# arima_data_resample = arima_df.resample('M').mean()                 #1month last day resample -downsample


##CORRELATION
df.corr()
pd.plotting.scatter_matrix(df)


# X['workClass'] = X['workClass'].astype('category')
# X['workClasscode'] = X['workClass'].cat.codes

# train.drop(columns = ['native-country'], inplace= True)
# train['workClass'] = train['workClass'].astype('categorical')
# train.replace('United-States', 'US', inplace = True)
# train.replace('<=50K', 'LessThan50K', inplace = True)
# train.replace('>50K', 'MoreThan50K', inplace = True)
# train.set_value('United-States', 'US')
# w = train['workClass'].value_counts()
# w.plot( kind = 'bar')
