from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns= iris.feature_names)
y = pd.DataFrame(iris.target, columns= ['target'])

# ['sepal length (cm)',
#  'sepal width (cm)',
#  'petal length (cm)',
#  'petal width (cm)']

f1 = X['sepal length (cm)']
f2 = X['sepal width (cm)']

plt.scatter(f1,f2)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

kmeans = KMeans(n_clusters= 2)
kmeans.fit(X)

y_kmeans = kmeans.predict(X)

plt.scatter(X['sepal length (cm)'], X['sepal width (cm)'], c=y_kmeans, s=50)
