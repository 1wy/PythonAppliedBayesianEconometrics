#-*- coding:utf-8 -*-
# author:wy
# datetime:19-1-4 下午12:14
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import numpy as np
from numpy.linalg import pinv, inv, cholesky
from numpy import matmul
from numpy.random import multinomial, dirichlet
from scipy.stats import invgamma, invwishart
from scipy.io import loadmat
import pandas as pd
from copy import deepcopy
# from MarkovSwitchingModel.utils import simS, hamilton_filter, switchg, posterior_para
from utils import simS, hamilton_filter, switchg, posterior_para
import matplotlib.pyplot as plt

T = 500
N = 2
B = np.array([[[0.2, 1]], [[0.9, -1]]])
sigma = np.array([[[3]], [[1]]])
P = np.array([[0.95, 0.05],[0.05, 0.95]])
strue = np.zeros((T,2))
strue[0,0] = 1
strue = simS(strue, P)

e = np.random.randn(T,1)
Y = np.zeros((T,1))
X = np.zeros((T,1))

for i in range(1,T):
	X[i,:] = Y[i-1,:]
	if strue[i,0] == 1:
		Y[i] = np.dot([X[i,:], 1], B[0].T) + e[i]*np.sqrt(sigma[0])
	else:
		Y[i] = np.dot([X[i, :], 1], B[1].T) + e[i] * np.sqrt(sigma[1])

y = pd.DataFrame(Y)
x = pd.DataFrame(X)
x.insert(len(x.columns), len(x.columns), 1)

phi = np.array([[[0.5], [1]],[[0.8], [-1]]])
sigs = np.array([[[3.]], [[1.]]])
p = 0.95
q = 0.95
pmat = np.array([[p, 1-q], [1-p, q]])
ncrit = 10

B0 = np.zeros((2,1))
sigma0 = np.eye(2)

d0 = 0.1
v0 = 1

u0 = np.array([25,5])
u1 = np.array([5,25])

out1 = []
out2 = []
out3 = []
out4 = []

reps = 10000
burn = 5000
cnt = 0
igibs = 0
while cnt < reps-burn:
	S = hamilton_filter(y.values, x.values, pmat, phi, sigs, ncrit)
	S = np.where(S == 1)[1]
	tranmat = switchg(S, N)
	# sample from dirichlet density
	p0 = dirichlet(tranmat[0,:] + u0, size=1).reshape(-1,1,order='F')
	p1 = dirichlet(tranmat[1,:] + u1, size=1).reshape(-1,1,order='F')
	pmat = np.hstack((p0,p1))


	for i in range(N):
		yi = y[S == i].values
		xi = x[S == i].values

		# sample beta
		M, V = posterior_para(yi, xi, sigs[i], B0, sigma0)
		phi[i] = M + matmul(np.random.randn(1,N), cholesky(V).T).T

		# sample sigma
		ei = yi - matmul(xi, phi[i])
		Ti = v0 + len(ei)
		Di = d0 + matmul(ei.T, ei)
		sigs[i] = invwishart.rvs(Ti, scale=Di)

	if igibs > burn-1:
		chck = phi[0,1,0] > phi[1,1,0] # constant bigger in regime 1
		if chck:
			tmp = np.array([n.flatten('F') for n in phi])
			out1.append(tmp.flatten(order='C'))
			out2.append(deepcopy(sigs).flatten())
			out3.append(S)
			out4.append([p0[0,0],p1[1,0]])
			cnt += 1

	igibs += 1
	print('replication %d, %d saved Draws' % (igibs, cnt))

B = np.squeeze(B)
plt.rc('font',size=10)
fig = plt.figure(figsize=(20,20))

plt.subplot(5,2,1)
plt.hist(np.array(out1)[:,0], 50)
plt.axvline(B[0,0], color = 'r')
plt.title('Coefficient regime 1')

plt.subplot(5,2,2)
plt.hist(np.array(out1)[:,2], 50)
plt.axvline(B[1,0], color = 'r')
plt.title('Coefficient regime 2')

plt.subplot(5,2,3)
plt.hist(np.array(out1)[:,1], 50)
plt.axvline(B[0,1], color = 'r')
plt.title('Intercept regime 1')

plt.subplot(5,2,4)
plt.hist(np.array(out1)[:,3], 50)
plt.axvline(B[1,1], color = 'r')
plt.title('Intercept regime 2')

plt.subplot(5,2,5)
plt.hist(np.array(out2)[:,0], 50)
plt.axvline(sigma[0], color = 'r')
plt.title('$\sigma_1$')

plt.subplot(5,2,6)
plt.hist(np.array(out2)[:,1], 50)
plt.axvline(sigma[1], color = 'r')
plt.title('$\sigma_2$')

plt.subplot(5,2,7)
plt.hist(np.array(out4)[:,0], 50)
plt.axvline(P[0,0], color = 'r')
plt.title('P00')

plt.subplot(5,2,8)
plt.hist(np.array(out4)[:,1], 50)
plt.axvline(P[1,1], color = 'r')
plt.title('P11')

plt.subplot(5,1,5)
tmp = np.mean(out3, axis=0)
plt.plot(tmp, 'c', linewidth=2)
plt.plot(strue[:,1], 'k', linewidth=2)
plt.title('probability of regime 1')
plt.legend(['estimate', 'true'])
plt.savefig('figures/example3.pdf')

