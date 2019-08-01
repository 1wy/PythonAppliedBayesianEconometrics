#-*- coding:utf-8 -*-
# author:wy
# datetime:18-12-27 上午9:54
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018


import numpy as np
from numpy import matmul
from numpy.linalg import inv, cholesky
import matplotlib.pyplot as plt

t = 500
Q = np.array([[0.001]])
R = np.array([[0.01]])
F = np.array([[1]])
mu = np.array([0])
e1 = np.random.randn(t,1) * np.sqrt(R)
e2 = np.random.randn(t,1) * np.sqrt(Q)
beta = np.zeros((t,1))
Y = np.zeros((t,1))
X = np.random.randn(t,1)

for j in range(1,t):
	beta[j,:] = beta[j-1,:] + e2[j,:]
	Y[j] = matmul(X[j,:], beta[j,:].T) + e1[j]

beta0 = np.zeros((1,1))
p00 = [[1]]
beta_tt = []
p_tt = []

beta11 = beta0
p11 = p00

for i in range(t):
	x = X[i,:]
	# prediction
	beta10 = mu + matmul(beta11,F.T)
	p10 = matmul(matmul(F,p11), F.T) + Q
	eta = Y[i,:] - matmul(x,beta10.T).T
	feta = matmul(matmul(x,p10),x.T) + R
	# update
	K = matmul(matmul(p10,x.T), inv(feta))
	beta11 = (beta10.T + matmul(K,eta.T)).T
	p11 = p10 - matmul(K,matmul(x,p10))

	beta_tt.append(beta11)
	p_tt.append(p11)

beta2 = np.zeros((t,1))
wa = np.random.randn(t,1)

i = t-1
p00 = p_tt[i]
beta2[i,:] = beta_tt[i] + matmul(wa[i,:], cholesky(p00).T)

for i in range(t-2,-1,-1):
	pt = p_tt[i]
	Kstar = matmul(matmul(pt,F.T),inv(matmul(matmul(F,pt), F.T)+Q))
	bm = beta_tt[i] + matmul(Kstar, (beta2[i+1] - mu - matmul(beta_tt[i], F.T)).T).T
	pm = pt - matmul(matmul(Kstar, F), pt)
	beta2[i,:] = bm + matmul(wa[i,:], cholesky(pm).T)

beta_tt = np.array(beta_tt).reshape(-1,1,order='F')
plt.plot(np.hstack((beta_tt, beta2, beta)))
plt.legend(['Kalman filter estimated $\\beta_{t}$','Draw from H($\\beta_{t}$)', 'true $\\beta_{t}$'])
plt.savefig('figures/example2.pdf')