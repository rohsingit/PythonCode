import pandas as pd
import matplotlib.pyplot as plt

##MEASURE TIME OF SCRIPT
import time
start_time = time.time()
.......
print("Time taken by script = %s seconds ---" % (time.time() - start_time))

##READ TO/FROM CSV/TXT OR EXCEL
path = r"C:/filename.csv"
df = pd.read_csv(path, encoding='cp1252', sep = ';', low_memory= False, header = None)    #Encoding if default 'utf-8' not working
df = pd.read_csv(path)
df = pd.read_excel(path, sheet_name = "NameOfSheet")
column_content = list(df['column'])       #Take all rows for that column as a list

df.to_csv("name_of_file.csv", sep = '~')
df.to_excel("name_of_file.xlsx", merge_cells = False)       #Don't merge cells

#LOAD AS TIME SERIES
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
df = pd.read_csv("path", parse_dates= ['columnName'], index_col= 'columnName', date_parser= dateparse) #or index_col = [0]
csv = pd.read_csv(filepath, encoding= 'cp1252', parse_dates= [0], index_col= [0], sep = ';', date_parser= dateparse)
                           
##DATE AND INDEX MANIPULATION
##Convert time from UNIX to usual format
df['Time'] = pd.to_datetime(df['Timestamp'], unit = 's')
df.set_index('Time', inplace= True)
del df['Timestamp']

##CREATE DATAFRAME/COLUMN
df = {'Filename': [], 'Pump_Model': [], 'Serial_Number': [], 'Motor_Number': []}
table = pd.DataFrame(df)

df['New_column_name'] = ''

##INSERT INTO LIST/DICTIONARY
dictionary = {     #Create new dictionary
		"iphone" : 2007,
		"iphone 3G" : 2008,
	}
released["iphone 5S"] = 2013  #Add to dictionary

list = ['a', b']
list.insert(index, elment)    #Insert into list

#Preview data
print(df.head())
print(df.tail())
print(df.describe())
print(df['CF_13'].describe())
print(df.dtypes)
print(df.index)
len(array_name) or len (dataframe)


##CREATE LIST FROM EXCEL COLUMN
path = r'C:\Users\rohitsingh\filename.xlsx'
List = pd.read_excel(path)
List = list(List['Column_Name'])      #Creates a list with members of column name


##FIND NULLS/UNIQUES/VALUE_COUNTS/FILLNNA/DROPNA/DROP_DUPLICATES
df.isnull().sum()
uniqes = list(df['package reference'].unique())
df['Column_Name'].value_counts().plot(kind = 'bar')   #Or assign to variable --> value_count = ...
df = df.fillna(method= 'ffill')
	
df_pkg['Count_of'] = df_pkg.groupby('column_name')['column_name'].transform('count')
df1.groupby(['Product ClassificationType', 'DeskType'])['mx2 trade number'].size().unstack().plot(kind='bar',stacked=True, rot = 0)

df.drop_duplicates(inplace= True)
df = df.sort_index(axis=1)
df = df.sort_values(['Legs','Strategy_Group','counterparty short label','Leg1'], ascending=[True, False, True, True, True])
df.drop_duplicates(inplace= True)

##SELECT SUBSET FROM DATAFRAME USING CELL VALUES #ISIN/CONTAINS/GROUPBY/ILOC
subset8EK = temp[temp['strategy'].str.contains("8E")]

some_list = ('Value1', 'Value2', 'Value3')  #Create a list or tuple to use ISIN
some_list = ['Value1', 'Value2', 'Value3']  #Create a list or tuple to use ISIN
Subset_DataFrame = df.loc[df['Column_Name'].isin(some_list)
if (df['column'].iloc[x] in ['RPL_M', ' ', 'RPL'])): Do something
                          
w = df.groupby(['trn date', 'strategy', 'trade typology', 'Fmly|Grp|Typ|'])['trn date'].count()                          

##DataFrame can be sliced using df.loc or df.iloc    df.ix is deprecated
#loc gets rows or columns with labels in the index
#iloc gets rows or columns with positions in the index
                          
df.iloc[0:4]        #Show rows 0 to 3
df.iloc[0:4,1]      #Show rows 0 to 3 from column 1
df.iloc[-5:]        #Show last 5 rows
df.iloc[:,0]        #Show all rows column 0
df.iloc[0,]         #Show first row all columns

df.loc['a']           #Get row whose index is 'a'
df.loc['b':, 'date']  #Get all rows from 'b' for column name 'date'
                          
subset = df.loc[df['Column_Name'] == 'Value']
subset = df.loc[(df['family'] == 'value') & (df['group'] == 'value2')]
subset = temp.loc[(temp['product family'] == 'IRD') & (temp['product group'].isin(group))]
subset.reset_index(drop=True, inplace = True)			  
 
Cell_value = subset.iloc[row_number]['Column_Name']

list = ['one', 'two', 'three']
if ((df['package typology'].iloc[j] in list) or (df['contract typology'].iloc[j] in list)):
    df['ClassificationType'].iloc[j] = 'FX'

                          
#MANIPULATE COLUMNS/CELLS/COLUMN TYPE -> REPLACE/SET_VALUE/ASTYPE/
del df['column_to_delete']                          
list(df)            #List column names
df = df.rename(columns={'oldName1': 'newName1', 'oldName2': 'newName2'})

col_my_order = ['C','B','A']
df = df[col_my_order]       #Rearrange columns in order
                         
LegName = 'Leg' + str(g+1)
subset[LegName] = subset.iloc[row_number]['column_name']                           
                          
X['workClass'] = X['workClass'].astype('category')
train['workClass'] = train['workClass'].astype('categorical')
train.replace('>50K', 'MoreThan50K', inplace = True)
train.set_value('United-States', 'US')                         
                          
#CONCAT, SET INDEX, APPEND
arima_df = pd.concat((timestamp, cf_100), axis = 1)
arima_df.set_index('Timestamp', inplace = True)
data = data.append(subset.iloc[0,])

#CHECK ARRAYS
np.array_equal(array1 , array2) #True or False
                           
##IF_ELSE/FOR
if ((df.iloc[i]['trn fmly'] == 'EQD') and (df.iloc[i]['trn group'] == 'EQS')):
        df.at[i, 'Vanilla or Complex'] = 'Complex'

for i in range(len(df)):
    print(i)

##PLOT USING PANDAS OR MATPLOTLIB
df.plot()                 #Plot everything
df.plot(y = 'column_name')       #Plot a line chart of a column using pandas' plot method
df.plot.box(rot =  90)

plt.plot(df['3852'])
plt.plot(df['3852'][0:10000])
plt.title("CF_32 plot")
plt.xticks(rotation=90)
plt.show()
plt.plot(df)
plt.axhline(0.55, color = 'red')      #Put after plotting to avoid x-axis rescaling                                                

df1.groupby(['Product ClassificationType', 'DeskType'])['mx2 trade number'].size().unstack().plot(kind='bar',stacked=True, rot = 0)
                          
###SUBPLOTTING
fig = plt.figure(1)
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(0, len(col_my_order1)):
    print("plotting:", col_my_order1[i])
    fig.add_subplot(2, 4, i+1)
    plt.title(col_my_order1[i])
    plt.xticks(rotation=90)
    plt.plot(df[col_my_order1[i]])
    plt.axhline(y=col_my_order1_limits[i], color='red')
plt.show()


#####PLOT ROLLING OR RESAMPLE MEAN##############

arima_df.rolling('3600s).mean().plot(title = "Hourly Avg")     #Hourly avg
arima_df.rolling('86400s').mean().plot(title = "Daily Avg")    #Daily avg
arima_df.rolling('604800s').mean().plot(title = "Weekly Avg")         #Weekly
arima_df.rolling('2592000s').mean().plot(title = "30-day Avg")   #30-day avg
arima_data = arima_df.rolling(window='2592000s').mean()

arima_data_resample = arima_df.resample('2592000s').mean()        #30-day resample -downsample
arima_data_resample = arima_df.resample('D').mean()                 #1day -downsample
arima_data_resample = arima_df.resample('M').mean()                 #1month last day resample -downsample


##CORRELATION
df.corr()
pd.plotting.scatter_matrix(df)
