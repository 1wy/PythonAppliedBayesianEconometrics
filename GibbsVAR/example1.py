#-*- coding:utf-8 -*-
# author:wy
# datetime:18-12-19 下午12:21
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import pandas as pd
from scipy.stats import invwishart
import numpy as np
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
	b0 = np.matmul(np.linalg.inv(np.matmul(x.T, x)), np.matmul(x.T, y))
	e = y - np.matmul(x,b0)
	s_i = np.sqrt(np.matmul(e.T, e) / (T - 2))
	s.append(s_i)

# specify parameters of the minnesota prior
lam1 = 1 # control the std of the prior on own lags, such as b_11, b_22
lam2 = 1 # control the std of the prior on lags of variables other than dependent variable, such as b_12, b_21
lam3 = 1 # control the degree to which coefficients on lags higher than 1 are likely to be zero
lam4 = 1 # control constant

# specify the prior mean of the coefficients of the Two equations of the VAR
B01 = np.reshape([0,1,0,0,0], (-1,1))
B02 = np.reshape([0,0,1,0,0], (-1,1))
B0 = np.vstack((B01, B02))

# Specify the prior variance of vec(B)
H = np.zeros((len(B0), len(B0)))
H[0,0] = (s[0] * lam4)**2
H[1,1] = lam1**2
H[2,2] = (s[0]*lam1*lam2/s[1])**2
H[3,3] = (lam1 / (np.power(2,lam3)))**2
H[4,4] = ((s[0]*lam1*lam2) / (s[1]*np.power(2,lam3)))**2
H[5,5] = (s[1]*lam4)**2
H[6,6] = (s[1]*lam1*lam2/s[0])**2
H[7,7] = lam1**2
H[8,8] = (s[1]*lam1*lam2 / (s[0]*np.power(2,lam3)))**2
H[9,9] = (lam1 / np.power(2,lam3))**2

# prior scale matrix for sigma the VAR covariance
S = np.eye(N)
alpha = N + 1 # prior degree of freedom

sigma = np.eye(N)
betaols = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y)).reshape(-1,1,order='F')

reps = 10000
burn = 5000
out1 = []
out2 = []

for j in range(reps):
	M_left = np.linalg.inv(np.linalg.inv(H) + np.kron(np.linalg.inv(sigma), np.matmul(X.T,X)))
	M_right = np.matmul(np.linalg.inv(H), B0) + np.matmul(np.kron(np.linalg.inv(sigma), np.matmul(X.T,X)), betaols)
	M = np.matmul(M_left, M_right)
	V = M_left

	beta = M + np.matmul(np.random.randn(1,len(B0)), np.linalg.cholesky(V).T).T
	e = Y - np.matmul(X, beta.reshape(N*L+1,N,order='F'))
	scale = np.matmul(e.T, e) + S
	sigma = invwishart.rvs(df=T+alpha, scale=scale)

	if j > burn-1:
		y_hat = np.zeros((14,2)) # forecast GDP growth and inflation for 3 years
		y_hat[0:2,:] = Y.iloc[-2:,:].values

		for i in range(2,14):
			y_hat[i,:] = np.matmul(np.hstack(([1], y_hat[i-1,:], y_hat[i-2,:])), beta.reshape(-1,N,order='F')) + \
			             np.matmul(np.random.randn(1,N), np.linalg.cholesky(sigma).T)

		out1.append(np.append(Y.iloc[:,0].values, y_hat[2:,0]))
		out2.append(np.append(Y.iloc[:,1].values, y_hat[2:,1]))

TT = np.arange(1948.75,2014.25,0.25)
fig = plt.figure()
plt.subplot(1,2,1)
plt.plot(TT, np.percentile(out1, [50, 10, 20, 30, 70, 80, 90], axis=0).T)
plt.xlim([1995, 2015])
plt.title('GDP Growth')

plt.subplot(1,2,2)
plt.plot(TT, np.percentile(out2,[50, 10, 20, 30, 70, 80, 90],axis=0).T)
plt.xlim([1995, 2015])
plt.title('Inflation')
plt.legend(['Median Forecast','10th percentile','20th percentile','30th percentile','70th percentile','80th percentile','90th percentile'])
plt.savefig('figures/example1.pdf')
