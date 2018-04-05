import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#LOAD TEST AND TRAIN DATA
train_path = r"C:\Users\10154861\Desktop\Test\adult.data.txt"
train = pd.read_csv(train_path, header = None, sep=',\s', na_values=["?"])
train.columns = ["age", "workClass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "50K"]

test_path =  r"C:\Users\10154861\Desktop\Test\adult.test.txt"
test = pd.read_csv(test_path, header = None, sep=',\s', na_values=["?"])
test.columns = ["age", "workClass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "50K"]

#CLEAN THE DATA OF NAs
# print(train.isnull().sum())
train = train.dropna()

# print(test.isnull().sum())
test = test.dropna()

#VISUALIZE THE DATA
# pd.plotting.scatter_matrix(train)
#
group1 = train.loc[train['50K'] == '>50K']
group2 = train.loc[train['50K'] == '<=50K']
#
# group1['age'].plot.hist()
# group2['age'].plot.hist()
#
# group1['workClass'].value_counts().plot.pie()
# group2['workClass'].value_counts().plot.pie()
#
# group1['education'].value_counts().plot.bar()
# group2['education'].value_counts().plot.bar()
#
# group1['education-num'].plot.hist()
# group2['education-num'].plot.hist()
#
# group1['marital-status'].value_counts().plot.bar()
# group2['marital-status'].value_counts().plot.bar()
#
# group1['occupation'].value_counts().plot.bar()
# group2['occupation'].value_counts().plot.bar()
#
# group1['relationship'].value_counts().plot.bar()
# group2['relationship'].value_counts().plot.bar()
#
# group1['race'].value_counts().plot.bar()
# group2['race'].value_counts().plot.bar()
#
# group1['sex'].value_counts().plot.pie()
# group2['sex'].value_counts().plot.pie()
#
# group1['hours-per-week'].plot.hist()
# group2['hours-per-week'].plot.hist()
#
# group1['native-country'].value_counts().plot.bar()
# group2['native-country'].value_counts().plot.bar()


#ALLOCATE TEST AND TRAINING MATRICES
y_train = train["50K"]
X_train = train.loc[:, train.columns != '50K']
y_test = test["50K"]
X_test = test.loc[:, test.columns != '50K']

#Y-test is read as 50K. Have to remove the dot
y_test.replace("<=50K.", "<=50K", inplace= True)
y_test.replace(">50K.", ">50K", inplace= True)

#LabelEncode the data
X_train_dum = pd.get_dummies(X_train)
X_test_dum = pd.get_dummies(X_test)

#Build the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, min_samples_split=5, min_samples_leaf=5)

model.fit(X_train_dum, y_train)

# plt.plot(list(X_dum), model.feature_importances_)
imp = pd.DataFrame(list(X_train_dum), model.feature_importances_)
imp.to_csv("imp.csv")

#The total number of columns are not the same in train and test set due to dummy-ing
# Get missing columns in the training test
missing_cols = set(X_train_dum.columns ) - set(X_test_dum.columns)
print(missing_cols)

X_test_dum['native-country_Holand-Netherlands'] = 0


# Ensure the order of column in the test set is in the same order than in train set
# X_test_dum == X_test_dum[X_train_dum.columns]

print("RandomForestClassifier Prediction on Training Data = ", (model.score(X_train_dum, y_train)))
print("RandomForestClassifier Prediction on Training Data = ", (model.score(X_test_dum, y_test)))

from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test_dum)))

##Decision tree to help visualize
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train_dum, y_train)

print("DecisionTree Prediction on Training Data = ", (model.score(X_train_dum, y_train)))
print("DecisionTree Prediction on Training Data = ", (model.score(X_test_dum, y_test)))

#
import graphviz, sklearn
feature_names = list(X_train_dum)
dot_data = sklearn.tree.export_graphviz(model,feature_names= feature_names, class_names= [">50K", "<=50K"], out_file= None)
graph = graphviz.Source(dot_data)
graph.render("rohit")

