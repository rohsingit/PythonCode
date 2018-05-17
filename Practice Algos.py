###########################--------------MODELS----------------############################################
from sklearn.linear_model import LinearRegression       #1
from sklearn.linear_model import LogisticRegression     #2
from sklearn.tree import DecisionTreeClassifier         #3a
from sklearn.tree import DecisionTreeRegressor          #3b
from sklearn.svm import SVC                             #4a Support Vector Machine
from sklearn.svm import SVR                             #4b
from sklearn.naive_bayes import GaussianNB              #5
from sklearn.neighbors import KNeighborsClassifier      #6
from sklearn.cluster import KMeans                      #7
from sklearn.ensemble import RandomForestClassifier     #8a
from sklearn.ensemble import RandomForestRegressor      #8b
from sklearn.decomposition import PCA, TruncatedSVD     #9
from sklearn.neural_network import MLPClassifier        #10 Multilayer Perceptron
from sklearn.ensemble import GradientBoostingClassifier #11
from statsmodels.tsa.arima_model import ARIMA           #12

###########################--------------MODELS END----------------########################################

from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

###########################--------------DATESETS----------------###########################################
from sklearn.datasets import load_iris, load_diabetes, load_boston
iris = load_iris()
diabetes = load_diabetes()
boston = load_boston()
data = iris             #TO BE SPECIFIED

X = pd.DataFrame(data.data, columns= data.feature_names)
y = pd.DataFrame(data.target, columns = ['target'])
df = pd.concat((X, y), axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25)
###########################--------------DATESETS END----------------########################################

#Exploratory
# pd.plotting.scatter_matrix(diabetes_df)
# print(diabetes_df.describe())
# print(diabetes_df.dtypes)
#
# ################1: Linear Regression###############################
print("\n")

model = LinearRegression()
model.fit(X_train, y_train)                  #OR model.fit(X_train[['bmi']], y_train)
print("LinearRegression Prediction on Training Data = ", (model.score(X_train, y_train)))
print("LinearRegression Prediction on Test Data = ", (model.score(X_test, y_test)))
# print(model.coef_)
# print(model.intercept_)
print("LinearRegression Mean squared error = ", (mean_squared_error(y_test, model.predict(X_test))))
print("LinearRegression R2 = ", (r2_score(y_test, model.predict(X_test))))
#
# #Plot the coefficients
# coeff = pd.DataFrame(model.coef_[0], index= list(X_train))
# coeff.plot(kind = 'bar')

#Visualisation only for univariate analysis
# plt.scatter(X_test[['bmi']], y_test)
# plt.plot(X_test[['bmi']], diabetes_y_pred, color = 'blue')
# plt.show()
#
# # ################2: Logistic Regression###############################
# print("\n")
#
# model = LogisticRegression()
# model.fit(X_train, y_train)
# print("LogisticRegression Prediction on Training Data = ", (model.score(X_train, y_train)))
# print("LogisticRegression Prediction on Test Data = ", (model.score(X_test, y_test)))
# # print(model.coef_)
# # print(model.intercept_)
# print("LogisticRegression Mean squared error = ", (mean_squared_error(y_test, model.predict(X_test))))
# print("LogisticRegression R2 = ", (r2_score(y_test, model.predict(X_test))))
# #
# # ################3a: Decision Tree Classifier################################
# print("\n")
#
# model = DecisionTreeClassifier()
# model.fit(X_train, y_train)
# print("DecisionTreeClassifier Prediction on Training Data = ", (model.score(X_train, y_train)))
# print("DecisionTreeClassifier Prediction on Test Data = ", (model.score(X_test, y_test)))
#
# import graphviz
# # dot_data = tree.export_graphviz(model, feature_names=iris.feature_names, class_names= iris.target_names, out_file= None)
# graph = graphviz.Source(dot_data)
# # graph.render("iris")
#
# print("DecisionTreeClassifier Mean squared error = ", (mean_squared_error(y_test, model.predict(X_test))))
# print("DecisionTreeClassifier R2 = ", (r2_score(y_test, model.predict(X_test))))
#
# # ################3b: Decision Tree Regressor################################
# print("\n")
#
# model = DecisionTreeRegressor()
# model.fit(X_train, y_train)
# print("DecisionTreeRegressor Prediction on Training Data = ", (model.score(X_train, y_train)))
# print("DecisionTreeRegressor Prediction on Test Data = ", (model.score(X_test, y_test)))
# # print(model.coef_)
# # print(model.intercept_)
# print("DecisionTreeRegressor Mean squared error = ", (mean_squared_error(y_test, model.predict(X_test))))
# print("DecisionTreeRegressor R2 = ", (r2_score(y_test, model.predict(X_test))))
# #
# # ################4a: Support vector Machine Classifier########################
# print("\n")
#
# model = SVC()                                       #kernel= 'linear' or polynomial or sigmoid
# model.fit(X_train, y_train)
# print("SVC Prediction on Training Data = ", (model.score(X_train, y_train)))
# print("SVC Prediction on Test Data = ", (model.score(X_test, y_test)))
# # print(model.coef_)
# # print(model.intercept_)
# print("SVC Mean squared error = ", (mean_squared_error(y_test, model.predict(X_test))))
# print("SVC R2 = ", (r2_score(y_test, model.predict(X_test))))
#
# # ################4b: Support vector Machine Regressor########################
# print("\n")
#
# model = SVR()
# model.fit(X_train, y_train)
# print("SVR Prediction on Training Data = ", (model.score(X_train, y_train)))
# print("SVR Prediction on Test Data = ", (model.score(X_test, y_test)))
# # print(model.coef_)
# # print(model.intercept_)
# print("SVR Mean squared error = ", (mean_squared_error(y_test, model.predict(X_test))))
# print("SVR R2 = ", (r2_score(y_test, model.predict(X_test))))
#
# # ################5: Gaussian Naive Bayes########################
# print("\n")
#
# model = GaussianNB()
# model.fit(X_train, y_train)
# print("GaussianNB Prediction on Training Data = ", (model.score(X_train, y_train)))
# print("GaussianNB Prediction on Test Data = ", (model.score(X_test, y_test)))
# # print(model.coef_)
# # print(model.intercept_)
# print("GaussianNB Mean squared error = ", (mean_squared_error(y_test, model.predict(X_test))))
# print("GaussianNB R2 = ", (r2_score(y_test, model.predict(X_test))))
#
# # ################6: K Nearest Neighbors########################
# print("\n")
#
# model = KNeighborsClassifier()
# model.fit(X_train, y_train)
# print("KNeighborsClassifier Prediction on Training Data = ", (model.score(X_train, y_train)))
# print("KNeighborsClassifier Prediction on Test Data = ", (model.score(X_test, y_test)))
# # print(model.coef_)
# # print(model.intercept_)
# print("KNeighborsClassifier Mean squared error = ", (mean_squared_error(y_test, model.predict(X_test))))
# print("KNeighborsClassifier R2 = ", (r2_score(y_test, model.predict(X_test))))


# ################7: K Nearest Neighbors########################
# print("\n")
#
# model = KMeans(n_clusters = 2)
# model.fit(X_train)
# print("KMeans Prediction on Training Data = ", (model.score(X_train, y_train)))
# print("KMeans Prediction on Test Data = ", (model.score(X_test, y_test)))
# # print(model.coef_)
# # print(model.intercept_)
# print("KMeans Mean squared error = ", (mean_squared_error(y_test, model.predict(X_test))))
# print("KMeans R2 = ", (r2_score(y_test, model.predict(X_test))))
#
# colors = np.array(['red', 'green', 'blue', 'black', 'green', 'yellow', 'grey'])
#
# plt.subplot(2, 2, 1)
# plt.title('Before classification 1')
# plt.scatter(X_train['sepal length (cm)'], X_train['sepal width (cm)'], c=colors[y['target']], s=40)
# plt.subplot(2, 2, 2)
# plt.title('Before classification 2')
# plt.scatter(X_train['sepal length (cm)'], X_train['petal width (cm)'], c=colors[y['target']], s=40)
#
# y_pred = model.labels_
#
# plt.subplot(2, 2, 3)
# plt.title('After classification 1')
# plt.scatter(X_train['sepal length (cm)'], X_train['sepal width (cm)'], c=colors[y_pred], s=40)
# plt.subplot(2,2,4)
# plt.title('After classification 2')
# plt.scatter(X_train['sepal length (cm)'], X_train['petal width (cm)'], c=colors[y_pred], s=40)

# ################8a: Random Forest Classifier########################
print("\n")

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("RandomForestClassifier Prediction on Training Data = ", (model.score(X_train, y_train)))
print("RandomForestClassifier Prediction on Test Data = ", (model.score(X_test, y_test)))
# print(model.coef_)
# print(model.intercept_)
print("RandomForestClassifier Mean squared error = ", (mean_squared_error(y_test, model.predict(X_test))))
print("RandomForestClassifier R2 = ", (r2_score(y_test, model.predict(X_test))))

#Confusion matrix using scikitplot
import scikitplot as skplt
cm = skplt.metrics.plot_confusion_matrix(y_test, model.predict(X_test))
plt.show()
#
# #classification report
# print(classification_report(y_test, model.predict(X_test), target_names= ['0', '1', '2']))
#
# #importances plot
# importances = pd.DataFrame(model.feature_importances_, index= list(X_train))
# importances.plot(kind = 'bar', rot = 0)
# plt.show()
#
# #confusion matrix using sklearn.metrics
#
# confusion_matrix(y_test, model.predict(X_test), labels= [0,1,2])

# # # ################8b: Random Forest Regressor########################
# print("\n")
#
# model = RandomForestRegressor()
# model.fit(X_train, y_train)
#
# print("RandomForestClassifier Prediction on Training Data = ", (model.score(X_train, y_train)))
# print("RandomForestClassifier Prediction on Test Data = ", (model.score(X_test, y_test)))
# # print(model.coef_)
# # print(model.intercept_)
# print("RandomForestClassifier Mean squared error = ", (mean_squared_error(y_test, model.predict(X_test))))
# print("RandomForestClassifier R2 = ", (r2_score(y_test, model.predict(X_test))))

# ################9: PCA########################
# print("\n")

#Before PCA score
# model = LinearRegression()
# model.fit(X_train, y_train)
# print("Score with LinearReg: " ,(model.score(X_test, y_test)))

# model = PCA(n_components= 8)
# pca_matrix = model.fit(X).transform(X)
#
# #After PCA score
# X_train, X_test, y_train, y_test = train_test_split(pca_matrix, y)
# model = LinearRegression()
# model.fit(X_train, y_train)
# print("Score with PCA: " ,(model.score(X_test, y_test)))

#Try Truncated SVD
# model = TruncatedSVD(n_components= 8)
# svd_matrix = model.fit(X).transform(X)
# X_train, X_test, y_train, y_test = train_test_split(svd_matrix, y)
# model = LinearRegression()
# model.fit(X_train, y_train)
# print("Score with SVD: " ,(model.score(X_test, y_test)))


# ################12: ARIMA########################

# arima_df = df['CF_4000']
# arima_data.plot()
# pd.plotting.autocorrelation_plot(arima_data_resample)

# plt.plot(arima_data, label = 'Original_dataset_CF_4000')
# model = ARIMA(arima_data, order = (2,1,2))         #p = AR,d = differences, q = MA
# results = model.fit()
# plt.plot(results.fittedvalues, color = 'red', label = 'forecasted_CF_4000')
# plt.legend()
# plt.show()

# # #do the Dicky-Fuller Test to check for Stationarity
# # from statsmodels.tsa.stattools import adfuller
# # print("Results of Dicky-Fuller Test: ")
# # dftest = adfuller(df_Vibration, autolag= 'AIC')
# # dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', '# lags Used', 'Number of Observations used'])
# # for key, value in dftest[4].items():
# #     dfoutput['Critical Value (%s)'%key] = value
# # print(dfoutput)

#########################EXTRA#################

# # from statsmodels.tsa.seasonal import seasonal_decompose
# # decomposition = seasonal_decompose(df_Vibration)
# #
# # from statsmodels.tsa.stattools import acf, pacf
# # acf_lag = acf(df_Vibration)
# # plt.plot(acf_lag)
# #
# # pacf_lag = pacf(df_Vibration)
# # plt.plot(pacf_lag)
# #
# plt.plot(df_Vibration, label = 'Original_dataset')
#
# model = ARIMA(train, order = (2,0,2))
# results = model.fit()
# plt.plot(results.fittedvalues, color = 'red', label = 'forecasted_training_set')
#
# plt.legend()
# plt.show()
# #
# forecast = results.forecast(steps= 78093)[0]
# forecast = pd.DataFrame(forecast, columns= ['forecasted value from test set'])
#
# rsme = sqrt(mean_squared_error(train,results.fittedvalues))
# print("RSME of training data vs itself = ", +rsme)
# rsme = sqrt(mean_squared_error(test, forecast))
# print("RSME of test data with forecasted data = ", +rsme)














####################PICKLING######################

################PICKLE A ML MODEL HERE################3

#Creating the pickle here

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import cloudpickle as pickle
import requests, json
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y)
rfc = RandomForestClassifier(n_estimators= 100, n_jobs= 2)
rfc.fit(X_train, y_train)

print(classification_report(y_test, rfc.predict(X_test)))

pickle.dump(rfc, open("iris_rfc.pkl", "wb"))

my_random_forest = pickle.load(open("iris_rfc.pkl", "rb"))
predict_values = [[5.84, 3.0, 3.75, 1.1]]
predicted = my_random_forest.predict(predict_values)
print(predicted)

