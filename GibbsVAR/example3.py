#-*- coding:utf-8 -*-
# author:wy
# datetime:18-12-20 上午11:04
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import pandas as pd
from scipy.stats import invwishart
from scipy.io import loadmat
import numpy as np
from numpy.linalg import inv
from numpy import matmul, kron
import matplotlib.pyplot as plt
from copy import deepcopy

data = pd.read_excel('data/datain.xls', header=None)
N = data.shape[1]
L = 2 # number of lags in the VAR

Y = data.iloc[L:,:]
X = pd.concat([data.shift(i+1) for i in range(L)], axis=1).dropna()
X.insert(0,'constant',1)
T = len(Y)

# compute standard deviation of each series residual via an ols regression
# to be used in setting the prior
s = []
for i in range(N):
	y = Y.iloc[:,[i]].values
	x = X.iloc[:,[0,i+1]].values
	b0 = matmul(inv(matmul(x.T, x)), matmul(x.T, y))
	e = y - matmul(x,b0)
	s_i = np.sqrt(matmul(e.T, e) / (T - 2))
	s.append(s_i)

# specify parameters of the minnesota prior
lam1 = 1 # control the std of the prior on own lags, such as b_11, b_22
lam2 = 1 # control the std of the prior on lags of variables other than dependent variable, such as b_12, b_21
lam3 = 1 # control the degree to which coefficients on lags higher than 1 are likely to be zero
lam4 = 1 # control constant

# specify the prior mean of the coefficients of the Two equations of the VAR
B01 = np.reshape([1,0,0,0], (-1,1))
B02 = np.reshape([0,1,0,0], (-1,1))
B0 = np.vstack((B01, B02))

# Specify the prior variance of vec(B)
H = np.zeros((len(B0), len(B0)))
H[0,0] = lam1**2
H[1,1] = (s[0]*lam1*lam2/s[1])**2
H[2,2] = (lam1 / (np.power(2,lam3)))**2
H[3,3] = ((s[0]*lam1*lam2) / (s[1]*np.power(2,lam3)))**2
H[4,4] = (s[1]*lam1*lam2/s[0])**2
H[5,5] = lam1**2
H[6,6] = (s[1]*lam1*lam2 / (s[0]*np.power(2,lam3)))**2
H[7,7] = (lam1 / np.power(2,lam3))**2

# prior scale matrix for sigma the VAR covariance
S = np.eye(N)
alpha = N + 1 # prior degree of freedom

# set priors for the long run mean which is a N by 1 vector
M0 = [[1], [1]]
V0 = 0.001 * np.eye(N)

betaols = matmul(inv(matmul(X.T, X)), matmul(X.T, Y))
F = np.vstack((betaols[1:,:].T, np.eye(N*(L-1), N*L)))
C = np.zeros((len(F),1)) # constant
C[:N] = betaols[[0],:].T

MU = matmul(inv(np.eye(len(F)) - F), C)
e = Y - matmul(X, betaols)
sigma = matmul(e.T, e) / T

reps = 10000
burn = 0

out1 = []
out2 = []

for j in range(reps):
	Y0 = pd.DataFrame(data.values - MU[:N,:].T)
	X0 = pd.concat([Y0.shift(i + 1) for i in range(L)], axis=1).dropna()
	Y0 = Y0.iloc[L:,:]

	bols = matmul(inv(matmul(X0.T, X0)), matmul(X0.T, Y0)).reshape(-1,1,order='F')
	M_left = inv(inv(H) + kron(inv(sigma), matmul(X0.T, X0)))
	M_right = matmul(inv(H), B0) + matmul(kron(inv(sigma), matmul(X0.T, X0)), bols)
	M = matmul(M_left, M_right)
	V = M_left

	beta = M + matmul(np.random.randn(1, len(B0)), np.linalg.cholesky(V).T).T
	e = Y0 - matmul(X0, beta.reshape(-1, N, order='F'))
	scale = matmul(e.T, e) + S
	sigma = invwishart.rvs(df=T + alpha, scale=scale)

	Y1 = Y.values - matmul(X.iloc[:,1:].values, beta.reshape(-1,N,order='F'))
	B_split = np.vsplit(beta.reshape(-1, N, order='F'), L)
	B_v = np.vstack(tuple([B_i.T for B_i in B_split]))
	U = np.vstack((np.eye(N), B_v))
	D = np.ones((T,L+1))
	D[:,1:] = -D[:,1:]

	Vstar = inv(inv(V0) + matmul(matmul(U.T, kron(matmul(D.T,D), inv(sigma))), U))
	Mstar = matmul(Vstar, matmul(U.T, matmul(matmul(inv(sigma), Y1.T), D).reshape(-1,1,order='F')) + matmul(inv(V0), M0))
	MU = Mstar + matmul(np.random.randn(1,N), np.linalg.cholesky(Vstar).T).T
	if j > burn-1:
		F = np.vstack((beta.reshape(-1,N,order='F').T, np.eye(N*(L-1), N*L)))
		mu = np.tile(MU,[L,1])

		C = matmul(np.eye(len(F)) - F, mu)

		y_hat = np.zeros((44,N))
		y_hat[:L,:] = Y.iloc[-L:,:].values
		for i in range(L, 44):
			y_hat[i,:] = C[:N].T + matmul(np.hstack((y_hat[i-1,:], y_hat[i-2,:])), beta.reshape(-1,N,order='F')) + \
			             matmul(np.random.randn(1,N), np.linalg.cholesky(sigma).T)

		out1.append(np.append(Y.iloc[:,0].values, y_hat[L:,0]))
		out2.append(np.append(Y.iloc[:,1].values, y_hat[L:,1]))


TT = np.arange(1948.75,2021.75,0.25)
fig = plt.figure()
plt.subplot(1,2,1)
plt.plot(TT, np.percentile(out1, [50, 10, 20, 30, 70, 80, 90], axis=0).T)
plt.xlim([1995, 2022])
plt.ylim([-6, 6])
plt.title('GDP Growth')

plt.subplot(1,2,2)
plt.plot(TT, np.percentile(out2,[50, 10, 20, 30, 70, 80, 90],axis=0).T)
plt.xlim([1995, 2022])
plt.ylim([-4, 6])
plt.title('Inflation')
# plt.legend(['Median Forecast','10th percentile','20th percentile','30th percentile','70th percentile','80th percentile','90th percentile'])
plt.savefig('figures/example3.pdf')


