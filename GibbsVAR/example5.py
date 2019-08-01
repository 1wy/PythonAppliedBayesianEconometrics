#-*- coding:utf-8 -*-
# author:wy
# datetime:18-12-21 下午3:21
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018


import pandas as pd
from scipy.stats import invwishart
from scipy.io import loadmat
import numpy as np
from GibbsVAR import create_dummies, check_right
from numpy.linalg import inv, cholesky
from numpy import matmul, kron
import matplotlib.pyplot as plt
from copy import deepcopy

data = pd.read_excel('data/usdata1.xls', header=None)
N = data.shape[1]
L = 2 # number of lags in the VAR

Y = data.iloc[L:,:]
X = pd.concat([data.shift(i+1) for i in range(L)], axis=1).dropna()
X.insert(len(X.columns),'constant',1)
X.columns = range(len(X.columns))
T = len(Y)

lamP = 1
tauP = 10*lamP
epsilonP = 1
muP = Y.mean()
sigmaP = []
deltaP = []

for i in range(N):
	y_temp = Y.iloc[:,[i]]
	x_temp = y_temp.shift(1)
	x_temp.insert(len(x_temp.columns),'constant',1)
	y_temp = y_temp.iloc[1:,:]
	x_temp = x_temp.iloc[1:,:]

	b_temp = matmul(inv(matmul(x_temp.T, x_temp)), matmul(x_temp.T,y_temp))
	e_temp = y_temp - matmul(x_temp,b_temp)
	s_temp = matmul(e_temp.T, e_temp) / len(y_temp)
	deltaP.append(np.squeeze(b_temp)[0])
	sigmaP.append(s_temp[0,0])

yd, xd = create_dummies(lamP, tauP, deltaP, epsilonP, L, muP, sigmaP, N)

Y0 = Y.append(pd.DataFrame(yd), ignore_index=True)
X0 = X.append(pd.DataFrame(xd), ignore_index=True)

betahat = matmul(inv(matmul(X0.T, X0)), matmul(X0.T, Y0))
xx = matmul(X0.T, X0)
ixx = inv(xx)
Mstar = betahat.reshape(-1, 1, order='F')

sigma = np.eye(N)

reps = 5000
burn = 3000
out = np.zeros((reps-burn, 36, N))
jj = 0
for j in range(reps):
	Vstar = kron(sigma, ixx)
	# randnum = loadmat('/home/wy/cslt/AppliedBayesianEconometrics/CHAPTER2/randnum.mat')['randnum']
	# beta = Mstar + matmul(randnum, cholesky(Vstar).T).T
	beta = Mstar + matmul(np.random.randn(1,N*(N*L+1)), cholesky(Vstar).T).T

	e = Y0 - matmul(X0, beta.reshape(-1,N,order='F'))
	scale = matmul(e.T, e)
	# sigma = loadmat('/home/wy/cslt/AppliedBayesianEconometrics/CHAPTER2/sigma.mat')['sigma']
	sigma = invwishart.rvs(df=T + len(yd), scale=scale)

	if j > burn-1:
		A0_hat =  cholesky(sigma).T
		while True:
			# K = loadmat('/home/wy/cslt/AppliedBayesianEconometrics/CHAPTER2/K.mat')['K']
			Q = np.linalg.qr(np.random.randn(N,N))[0]
			A0 = matmul(Q, A0_hat)

			symbol = np.array([1, -1, -1, -1, 1, -1, -1])
			if all(symbol * A0[0,[0,1,2,3,4,5,7]] > 0):
				break

			elif all(-1*symbol*A0[0,[0,1,2,3,4,5,7]] > 0):
				A0[0,[0,1,2,3,4,5,7]] = -A0[0,[0,1,2,3,4,5,7]]
				break

		y_hat = np.zeros((36, N))
		v_hat = np.zeros((36, N))
		v_hat[2,0] = 1
		for i in range(2, 36):
			y_hat[i, :] = matmul(np.hstack((y_hat[i - 1, :], y_hat[i - 2, :], [0])), beta.reshape(-1, N, order='F')) + \
			              matmul(v_hat[i,:], A0)

		out[jj,:,:] = y_hat
		jj += 1


fig = plt.figure()
titles = ['FederalFunds Rate', 'GDP Growth', 'CPI Inflation', 'PCE Growth', 'Unemployment', 'Investment',
          'Net Exports', 'M2', '10 year Government Bond Yield', 'Stock Price Growth', 'Yen Dollar Rate']
for i in range(1,12):
	plt.subplot(4,3,i)
	temp = out[:,:,i-1]
	temp = np.squeeze(np.percentile(temp, [50, 16, 84], axis=0).T)
	plt.plot(temp[2:,:])
	plt.title(titles[i-1], fontsize='small')

plt.savefig('figures/example5.pdf')

