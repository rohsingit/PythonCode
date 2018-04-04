import numpy as np
import pandas as pd
from scipy import stats


from sklearn.datasets import load_boston, load_iris
boston = load_boston()
iris = load_iris()

###CHOSSE THE DATA YOU WANT TO USE
data = boston

X = pd.DataFrame(data.data, columns= data.feature_names)
y = pd.DataFrame(data.target, columns= ['target'])

#####PLOTTING/VISUALIZING#######
# y.boxplot()

#####CORRELATION COEFFICIENT#####
# X['AGE'].corr(y['target'])



######T_TESTS#######


# N = 100     #Sample size
#
# a = np.random.randn(N) + 10
# b = np.random.randn(N)

#chi2# #
# t, p = stats.ttest_ind(X['AGE'],y)
# print(t , p)


# black = [9,10,12,11,8,10]
# red   = [6,5,14,15,11,9]
#
# chi2, p = stats.chisquare(black)
# print(chi2, p)

#Z-test


#ANOVA

# F, p = stats.f_oneway()