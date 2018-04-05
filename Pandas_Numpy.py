import pandas as pd
import matplotlib.pyplot as plt

##READ FILES/GENERATE DATA
#From CSV
path = r"C:\Users\10154861\Desktop\Work\1. Analytics\A. Data\A. VM_Holz_Biomasse\C. 4_yrs_complete\2985.csv"
df = pd.read_csv(path)
df.isnull().sum()

path = r"C:\Users\10154861\Desktop\Work\1. Analytics\A. Data\A. VM_Holz_Biomasse\A. 4_yrsData_IPS_Detect_iCMS\D1_P003_SIHI_MSC_065140_non_drive_side_2985_7.csv"
df2 = pd.read_csv(path)

path = r"C:\Users\10154861\Desktop\Work\1. Analytics\A. Data\A. VM_Holz_Biomasse\A. 4_yrsData_IPS_Detect_iCMS\D1_P003_SIHI_MSC_065140_drive_side_2990_6.csv"
df3 = pd.read_csv(path)

#Create DataFrame
# df = {'Filename': [], 'Pump_Model': [], 'Serial_Number': [], 'Motor_Number': []}
# table = pd.DataFrame(df)

##Preview data
# print(df.head())
# print(df.tail())
# print(df.describe())
# print(df.dtypes)
# print(df.index)

##Manipulate Columns
list(df)            #List column names


##Check for missing cells
# df.isnull().sum()

##PLOTTING/EXPLORING
# df.plot()             #Plot everything
# df.plot(y = '3852')     #Plot a line chart of a column using pandas' plot method
# df.boxplot(column = '3852')     #Plot a boxplot of a column
#
# # plt.plot(df['3852'])
# plt.plot(df['3852'][0:10000])
#
# # plt.axhline(0.55, color = 'red')
# # plt.title("CF_32 plot")
# #
# # plt.show()




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
