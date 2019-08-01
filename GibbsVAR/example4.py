#-*- coding:utf-8 -*-
# author:wy
# datetime:18-12-20 下午5:51
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import pandas as pd
from scipy.stats import invwishart
from scipy.io import loadmat
import numpy as np
from numpy.linalg import inv, cholesky
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

s = []
for i in range(N):
	y = Y.iloc[:,[i]].values
	x = X.iloc[:,[0,i+1]].values
	b0 = np.matmul(np.linalg.inv(np.matmul(x.T, x)), np.matmul(x.T, y))
	e = y - np.matmul(x,b0)
	s_i = np.squeeze(np.sqrt(np.matmul(e.T, e) / (T - 2)))
	s.append(s_i)

mu = Y.mean()
tau = 0.1
d = 1
lamc = 1.
lam = 1.
delta = 1.

yd1 = np.array([[s[0], 0],[0, s[1]]]) / tau
xd1 = np.array([[0, s[0], 0, 0, 0], [0, 0, s[1], 0, 0]]) / tau

yd2 = np.array([[0, 0],[0, 0]])
xd2 = np.array([[0, 0, 0, s[0]*np.power(2,d), 0], [0, 0, 0, 0, s[1]*np.power(2,d)]]) / tau

yd3 = np.array([[0, 0],[0, 0]])
xd3 = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]) / lamc

yd6 = np.array([[s[0], 0],[0, s[1]]])
xd6 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

yd4 = np.array([[mu[0], 0],[0, mu[1]]]) * lam
xd4 = np.array([[0, mu[0], 0, mu[0], 0], [0, 0, mu[1], 0, mu[1]]]) * lam

yd5 = np.array([[mu[0], mu[1]]]) * delta
xd5 = np.array([[1, mu[0], mu[1], mu[0], mu[1]]]) * delta

yd = np.vstack((yd1, yd2, yd3, yd4, yd5, yd6))
xd = np.vstack((xd1, xd2, xd3, xd4, xd5, xd6))

Ystar = pd.DataFrame(np.vstack((Y.values, yd)))
Xstar = pd.DataFrame(np.vstack((X.values, xd)))
Tstar = len(Xstar)

betahat = matmul(inv(matmul(Xstar.T, Xstar)), matmul(Xstar.T, Ystar))
e = Ystar - matmul(Xstar, betahat)
sigma = matmul(e.T, e) / T

reps = 10000
burn = 5000

out1 = []
out2 = []

for j in range(reps):

	M = betahat.reshape(-1,1,order='F')
	V = kron(sigma, inv(matmul(Xstar.T, Xstar)))

	beta = M + matmul(np.random.randn(1,N*(N*L+1)), cholesky(V).T).T
	e = Ystar - matmul(Xstar, beta.reshape(-1,N,order='F'))
	scale = matmul(e.T, e)

	sigma = invwishart.rvs(df=Tstar, scale=scale)
	if j>burn-1:
		y_hat = np.zeros((14, N))
		y_hat[:L, :] = Y.iloc[-L:, :].values
		for i in range(L, 14):
			y_hat[i, :] = matmul(np.hstack(([1], y_hat[i - 1, :], y_hat[i - 2, :])), beta.reshape(-1, N, order='F')) + \
			              matmul(np.random.randn(1, N), cholesky(sigma).T)

		out1.append(np.append(Y.iloc[:, 0].values, y_hat[L:, 0]))
		out2.append(np.append(Y.iloc[:, 1].values, y_hat[L:, 1]))

TT = np.arange(1948.75,2014.25,0.25)
fig = plt.figure()
plt.subplot(1,2,1)
plt.plot(TT, np.percentile(out1, [50, 10, 20, 30, 70, 80, 90], axis=0).T)
plt.xlim([1995, 2015])
plt.ylim([-6, 8])
plt.title('GDP Growth')

plt.subplot(1,2,2)
plt.plot(TT, np.percentile(out2,[50, 10, 20, 30, 70, 80, 90],axis=0).T)
plt.xlim([1995, 2015])
plt.ylim([-2, 6])
plt.title('Inflation')
# plt.legend(['Median Forecast','10th percentile','20th percentile','30th percentile','70th percentile','80th percentile','90th percentile'])
plt.savefig('figures/example4.pdf')
