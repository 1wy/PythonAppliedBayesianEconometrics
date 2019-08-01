#-*- coding:utf-8 -*-
# author:wy
# datetime:18-12-25 下午4:41
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import pandas as pd
from scipy.stats import invwishart
from scipy.io import loadmat
import numpy as np
from GibbsVAR import create_dummies, check_right, irfsim
from numpy.linalg import inv, pinv, cholesky
from scipy.linalg import sqrtm
from numpy import matmul, kron
import matplotlib.pyplot as plt
from copy import deepcopy

data = pd.read_excel('data/datain.xls', header=None)
N = data.shape[1]
L = 2 # number of lags in the VAR
horizon = 3
path = np.array([[1],[1],[1]])

Y = data.iloc[L:,:]
X = pd.concat([data.shift(i+1) for i in range(L)], axis=1).dropna()
X.insert(len(X.columns),'constant',1)
X.columns = range(len(X.columns))
T = len(Y)

B = matmul(inv(matmul(X.T, X)), matmul(X.T, Y))
e = Y - matmul(X,B)
sigma = matmul(e.T, e) / T
A0 = cholesky(sigma).T

S = np.zeros((1,N))
S[0,0] = 1
Z1 = irfsim(B, N, L, A0, S, horizon+L)
S = np.zeros((1,N))
S[0,1] = 1
Z2 = irfsim(B, N, L, A0, S, horizon+L)

y_hat1 = np.zeros((horizon + L, N))
y_hat1[:L,:] = Y.iloc[-L:,:]

for i in range(L,horizon+L):
	x = []
	for j in range(L):
		x = np.append(x, y_hat1[i-j-1,:])
	y_hat1[i,:] = matmul(np.hstack((x,[1])), B)

y_hat1 = y_hat1[L:,:]
R = np.array([[Z1[0,1], Z2[0,1], 0, 0, 0, 0],
              [Z1[1,1], Z2[1,1], Z1[0,1], Z2[0,1], 0, 0],
              [Z1[2,1], Z2[2,1], Z1[1,1], Z2[1,1], Z1[0,1], Z2[0,1]]])

r = path - y_hat1[:,[1]]
e_hat = matmul(matmul(R.T, pinv(matmul(R,R.T))), r)
e_hat = e_hat.reshape(N,horizon,order='F').T

y_hat2 = np.zeros((horizon+L,N))
y_hat2[:L,:] = Y.iloc[-L:,:]

for i in range(L,horizon+L):
	x = []
	for j in range(L):
		x = np.append(x, y_hat2[i-j-1,:])
	y_hat2[i,:] = matmul(np.hstack((x,[1])), B) + matmul(e_hat[i-L,:], A0)

y_hat2 = y_hat2[L:,:]

reps = 5000
burn = 3000
out1 = []
out2 = []
y_hatg = y_hat2
sig = sigma

for igibbs in range(reps):
	datag = data.append(pd.DataFrame(y_hatg))
	Ystar = datag.iloc[L:,:]
	Xstar = pd.concat([datag.shift(i+1) for i in range(L)], axis=1).dropna()
	Xstar.insert(len(Xstar.columns), 'constant', 1)
	Xstar.columns = range(len(Xstar.columns))
	T = len(Xstar)

	M = matmul(inv(matmul(Xstar.T, Xstar)), matmul(Xstar.T, Ystar)).reshape(-1,1,order='F')
	V = kron(sig, inv(matmul(Xstar.T, Xstar)))
	bg = M + matmul(np.random.randn(1, N*(N*L+1)), cholesky(V).T).T
	bg1 = bg.reshape(-1,N,order='F')

	e = Ystar - matmul(Xstar, bg1)
	scale = matmul(e.T, e)
	sig = invwishart.rvs(df=T, scale=scale)

	A0g = cholesky(sig).T


	S = np.zeros((1, N))
	S[0, 0] = 1
	Z1 = irfsim(bg1, N, L, A0g, S, horizon + L)
	S = np.zeros((1, N))
	S[0, 1] = 1
	Z2 = irfsim(bg1, N, L, A0g, S, horizon + L)

	y_hat1 = np.zeros((horizon + L, N))
	y_hat1[:L, :] = Y.iloc[-L:, :]

	for i in range(L,horizon+L):
		x = []
		for j in range(L):
			x = np.append(x, y_hat1[i-j-1,:])
		y_hat1[i,:] = matmul(np.hstack((x,[1])), bg1)

	y_hat1 = y_hat1[L:,:]
	R = np.array([[Z1[0,1], Z2[0,1], 0, 0, 0, 0],
	              [Z1[1,1], Z2[1,1], Z1[0,1], Z2[0,1], 0, 0],
	              [Z1[2,1], Z2[2,1], Z1[1,1], Z2[1,1], Z1[0,1], Z2[0,1]]])

	r = path - y_hat1[:, [1]]
	MBAR = matmul(matmul(R.T, pinv(matmul(R, R.T))), r)
	VBAR = matmul(matmul(R.T, pinv(matmul(R, R.T))), R)
	VBAR = np.eye(len(VBAR)) - VBAR

	e_draw = MBAR + matmul(np.random.randn(1,len(MBAR)), np.real(sqrtm(VBAR))).T
	e_draw = e_draw.reshape(N, horizon, order='F').T

	y_hatg = np.zeros((horizon + L, N))
	y_hatg[:L, :] = Y.iloc[-L:, :]

	for i in range(L, horizon + L):
		x = []
		for j in range(L):
			x = np.append(x, y_hatg[i-j-1, :])
		y_hatg[i, :] = matmul(np.hstack((x, [1])), bg1) + matmul(e_draw[i-L, :], A0g)
	y_hatg = y_hatg[L:, :]

	if igibbs > burn-1:
		out1.append(np.append(Y.iloc[:,0].values, y_hatg[:,0]).T)
		out2.append(np.append(Y.iloc[:,1].values, y_hatg[:,1]).T)


TT = np.arange(1948.75,2012,0.25)
fig = plt.figure()
plt.subplot(1,2,1)
plt.plot(TT, np.percentile(out1, [50, 10, 20, 30, 70, 80, 90], axis=0).T)
plt.xlim([1995, np.max(TT)+0.25])
plt.title('GDP Growth')

plt.subplot(1,2,2)
plt.plot(TT, np.percentile(out2,[50, 10, 20, 30, 70, 80, 90],axis=0).T)
plt.xlim([1995, np.max(TT)+0.25])
plt.title('Inflation')
plt.legend(['Median Forecast','10th percentile','20th percentile','30th percentile','70th percentile','80th percentile','90th percentile'])
plt.savefig('figures/example8.pdf')
