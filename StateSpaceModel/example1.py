#-*- coding:utf-8 -*-
# author:wy
# datetime:18-12-26 下午9:07
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import numpy as np
from numpy import matmul
from numpy.linalg import inv
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

beta_tt = np.array(beta_tt).reshape(-1,1,order='F')
p_tt = np.array(beta_tt.reshape(-1,1,1,order='F'))
plt.plot(np.hstack((beta_tt, beta)))
plt.legend(['estimated $\\beta_{t}$','true $\\beta_{t}$'])
plt.savefig('figures/example1.pdf')