import numpy as np
from scipy import stats

# N = 100     #Sample size
#
# a = np.random.randn(N) + 10
# b = np.random.randn(N)
#
# t, p = stats.ttest_ind(a,b)
# print(t , p)

#chi2

black = [9,10,12,11,8,10]
red   = [6,5,14,15,11,9]

chi2, p = stats.chisquare(black)
print(chi2, p)

#Z-test


#ANOVA

F, p = stats.f_oneway()