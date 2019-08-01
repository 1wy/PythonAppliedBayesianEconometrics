#-*- coding:utf-8 -*-
# author:wy
# datetime:19-1-9 下午8:03
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import numpy as np
from numpy import matmul
from scipy.io import loadmat
from copy import deepcopy
from numpy.linalg import inv, cholesky
import matplotlib.pyplot as plt


T = 100
sigma = 1
B1 = 4
B2 = 2
e = np.random.randn(T,1) * np.sqrt(sigma)
X = np.random.rand(T,1)
Y = B1 * (X ** B2) + e

gammaOld = np.array([[0], [0], [0.1]])
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
	if sigma2 < 0:
		posteriorNew = -1000000
	else:
		em = Y - b1 * (X ** b2)
		lik = -(T/2)*np.log(2*np.pi*sigma2 ) - 0.5*matmul(em.T, em)/ sigma2
		posteriorNew = lik[0,0]

	b1, b2, sigma2 = tuple(gammaOld.flatten())
	em = Y - b1 * (X ** b2)
	lik = -(T / 2) * np.log(2 * np.pi*sigma2) - 0.5 * matmul(em.T, em) / sigma2
	posteriorOld = lik[0,0]

	accept = np.min([np.exp(posteriorNew - posteriorOld), 1])
	if np.random.randn() < accept:
		gammaOld = gammaNew
		naccept += 1

	out.append(gammaOld.flatten())

plt.plot(np.hstack((out, true)))
plt.xlabel('Metropolis Hastings Draws')
plt.legend(['$B_1$','$B_2$','$\sigma^2$','True $B_1$','True $B_2$','True $\sigma^2$'])
plt.savefig('figures/example1.pdf')
