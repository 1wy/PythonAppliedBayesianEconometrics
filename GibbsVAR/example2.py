#-*- coding:utf-8 -*-
# author:wy
# datetime:18-12-19 下午7:54
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import sys
import pandas as pd
from scipy.stats import invwishart
import numpy as np
from GibbsVAR import stability
import matplotlib.pyplot as plt
from copy import deepcopy

data = pd.read_excel('data/dataUS.xls', header=None)
N = data.shape[1]
L = 2 # number of lags in the VAR

Y = data.iloc[L:,:]
X = pd.concat([data.shift(i+1) for i in range(L)], axis=1).dropna()
X.insert(0,'constant',1)
T = len(Y)

s = []
for i in range(N):
	y = Y.iloc[:,[i]].values
	x = X.iloc[:,[0,i+1]].values
	b0 = np.matmul(np.linalg.inv(np.matmul(x.T, x)), np.matmul(x.T, y))
	e = y - np.matmul(x,b0)
	s_i = np.sqrt(np.matmul(e.T, e) / (T - 2))
	s.append(s_i)

lam1 = 0.1
lam3 = 0.05
lam4 = 1

B0 = np.hstack((np.zeros((N,1)), 0.95*np.eye(N), np.zeros((N,N)))).reshape(-1,1,order='C')
H = np.eye(len(B0))

H[2,2] = 1e-9
H[3,3] = 1e-9
H[4,4] = 1e-9
H[6,6] = 1e-9
H[7,7] = 1e-9
H[8,8] = 1e-9

H[0,0] = (s[0]*lam4)**2
H[1,1] = (lam1)**2
H[5,5] = (lam1/np.power(2,lam3))**2
# second equation
H[9,9] = (s[1]*lam4)**2
H[10,10] = ((s[1]*lam1)/s[0])**2
H[11,11] = (lam1)**2
H[12,12] = ((s[1]*lam1)/s[2])**2
H[13,13] = ((s[1]*lam1)/s[3])**2
H[14,14] = ((s[1]*lam1)/(s[0]*np.power(2,lam3)))**2
H[15,15] = (lam1/np.power(2,lam3))**2
H[16,16] = ((s[1]*lam1)/(s[2]*np.power(2,lam3)))**2
H[17,17] = ((s[1]*lam1)/(s[3]*np.power(2,lam3)))**2
# third equation
H[18,18] = (s[2]*lam4)**2
H[19,19] = ((s[2]*lam1)/s[0])**2
H[20,20] = ((s[2]*lam1)/s[1])**2
H[21,21] = (lam1)**2
H[22,22] = ((s[2]*lam1)/s[3])**2
H[23,23] = ((s[2]*lam1)/(s[0]*np.power(2,lam3)))**2
H[24,24] = ((s[2]*lam1)/(s[1]*np.power(2,lam3)))**2
H[25,25] = (lam1/np.power(2,lam3))**2
H[26,26] = ((s[2]*lam1)/(s[3]*np.power(2,lam3)))**2
# fourth equation
H[27,27] = (s[3]*lam4)**2
H[28,28] = ((s[3]*lam1)/s[0])**2
H[29,29] = ((s[3]*lam1)/s[1])**2
H[30,30] = ((s[3]*lam1)/s[2])**2
H[31,31] = (lam1)**2
H[32,32] = ((s[3]*lam1)/(s[0]*np.power(2,lam3)))**2
H[33,33] = ((s[3]*lam1)/(s[1]*np.power(2,lam3)))**2
H[34,34] = ((s[3]*lam1)/(s[2]*np.power(2,lam3)))**2
H[35,35] = (lam1/np.power(2,lam3))**2

S = np.eye(N)
alpha = N+1

sigma = np.eye(N)
betaols = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y)).reshape(-1,1,order='F')

reps = 40000
burn = 30000
out1 = []
out2 = []
out3 = []
out4 = []

for j in range(reps):
	# draw beta
	M_left = np.linalg.inv(np.linalg.inv(H) + np.kron(np.linalg.inv(sigma), np.matmul(X.T,X)))
	M_right = np.matmul(np.linalg.inv(H), B0) + np.matmul(np.kron(np.linalg.inv(sigma), np.matmul(X.T,X)), betaols)
	M = np.matmul(M_left, M_right)
	V = M_left

	# draw sigma
	while True:
		beta = M + np.matmul(np.random.randn(1, len(B0)), np.linalg.cholesky(V).T).T
		# check the stability of the vars
		if stability(beta, N, L):
			break

	e = Y - np.matmul(X, beta.reshape(N * L + 1, N, order='F'))
	scale = np.matmul(e.T, e) + S
	sigma = invwishart.rvs(df=T+alpha, scale=scale)

	if j > burn-1:
		A0 = np.linalg.cholesky(sigma).T
		v = np.zeros((60,N))
		v[L,1] = -1
		y_hat = np.zeros((60, N))  # forecast GDP growth and inflation for 3 years
		for i in range(L,60):
			y_hat[i, :] = np.matmul(np.hstack(([0], y_hat[i - 1, :], y_hat[i - 2, :])),beta.reshape(-1, N, order='F')) + \
			              np.matmul(v[[i],:], A0)

		out1.append(y_hat[L:,0])
		out2.append(y_hat[L:,1])
		out3.append(y_hat[L:,2])
		out4.append(y_hat[L:,3])

fig = plt.figure()
TT = len(out1[0])

plt.subplot(2,2,1)
plt.plot(np.percentile(out1, [50, 16, 84], axis=0).T)
plt.hlines(0,0,TT, colors='r')
plt.title('Federal Funds rate')

plt.subplot(2,2,2)
plt.plot(np.percentile(out2, [50, 16, 84], axis=0).T)
plt.hlines(0,0,TT, colors='r')
plt.title('Government Bond Yield')

plt.subplot(2,2,3)
plt.plot(np.percentile(out3, [50, 16, 84], axis=0).T)
plt.hlines(0,0,TT, colors='r')
plt.title('Unemployment Rate')

plt.subplot(2,2,4)
plt.plot(np.percentile(out4, [50, 16, 84], axis=0).T)
plt.hlines(0,0,TT, colors='r')
plt.title('Inflation')
plt.legend(['Median Response','Upper 84%','Lower 16%'])

plt.savefig('figures/example2.pdf')
