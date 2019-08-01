#-*- coding:utf-8 -*-
# author:wy
# datetime:19-1-9 下午8:03
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import numpy as np
from numpy import matmul
from scipy.io import loadmat
from scipy.stats import multivariate_normal, gamma
from copy import deepcopy
from numpy.linalg import inv, cholesky, det
from MetropolisHasting.utils import *
import matplotlib.pyplot as plt


T = 100
sigma = 1
B1 = 4
B2 = 2
e = np.random.randn(T,1) * np.sqrt(sigma)
X = np.random.rand(T,1)
Y = B1 * (X ** B2) + e

gammaOld = np.array([[0], [0], [0.1]])

B0 = np.zeros(2)
sigma0 = np.eye(2) * 100
s0 = 1
v0 = 5

yols = deepcopy(Y)
xols = np.hstack((np.ones((T,1)), X))
bols = matmul(inv(matmul(xols.T, xols)), matmul(xols.T, yols))
eols = yols - matmul(xols, bols)
sols = matmul(eols.T, eols) / T
vols = sols * inv(matmul(xols.T, xols))

P = np.eye(3)
P[0,0] = vols[0,0]
P[1,1] = vols[1,1]
P[2,2] = 0.1

K = 0.2
P = K * P

reps = 15000
true = np.tile([[B1, B2, sigma]], (reps, 1))

out = []
naccept = 0

for j in range(reps):
	gammaNew = gammaOld + matmul(np.random.randn(1,3), cholesky(P).T).T
	b1, b2, sigma2 = tuple(gammaNew.flatten())
	if sigma2 <= 0:
		posteriorNew = -1000000
	else:
		b = [b1,b2]
		em = Y - b1 * (X ** b2)
		posteriorNew = log_posterior(em, b, sigma2, B0, sigma0, v0, s0, T)
	b1, b2, sigma2 = tuple(gammaOld.flatten())
	b = [b1,b2]
	em = Y - b1 * (X ** b2)
	posteriorOld = log_posterior(em, b, sigma2, B0, sigma0, v0, s0, T)
	accept = np.min([np.exp(posteriorNew - posteriorOld), 1])
	if np.random.rand() < accept:
		gammaOld = deepcopy(gammaNew)
		naccept += 1

	out.append(gammaOld.flatten())

plt.plot(np.hstack((out, true)))
plt.xlabel('Metropolis Hastings Draws')
plt.legend(['$B_1$','$B_2$','$\sigma^2$','True $B_1$','True $B_2$','True $\sigma^2$'])
plt.savefig('figures/example2.pdf')
