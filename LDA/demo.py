# -*- coding: utf-8 -*-

# Generating random data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.linalg as lin


x_train = np.array([[2, 2], [0, 0], [-1, 0], [-1, -2]])


y_train = np.array([1, 1, 2, 2])

n = len(y_train)

# Since K == 2, we have two different mu values
mu1 = np.mean(x_train[y_train == 1, :], axis=0)
mu2 = np.mean(x_train[y_train == 2, :], axis=0)

Sigma = np.zeros((2, 2))
for i in range(n):
    if y_train[i] == 1:
        Sigma += np.outer((x_train[i, :]-mu1), x_train[i, :]-mu1)/n
    else:
        Sigma += np.outer((x_train[i, :]-mu2), x_train[i, :]-mu2)/n
q1 = sum(y_train == (1))/n
q2 = sum(y_train == (2))/n

Sigma_Inv = lin.inv(Sigma)
Sigma_Sqroot = lin.sqrtm(Sigma)
eigen = lin.eig(Sigma)
print(Sigma_Inv)
