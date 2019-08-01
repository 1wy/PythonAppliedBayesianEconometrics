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
# from MarkovSwitchingModel.utils import *
from utils import *
import matplotlib.pyplot as plt

T = 500
N = 2
B = np.array([[[0.2, 1]], [[0.9, -1]]])
sigma = np.array([[[3.]], [[1.]]])
GAMMA = np.array([-1,1.])
LAM0 = 10

strue = np.zeros((T,1))
strue[0,0] = 1
Z = getar([0.9], T, 1)
strue, ptrue, qtrue = simS_varyingP(strue, Z, GAMMA, LAM0)

e = np.random.randn(T,1)
Y = np.zeros((T,1))
X = np.zeros((T,1))

for i in range(1,T):
	X[i,:] = Y[i-1,:]
	if strue[i,0] == 0:
		Y[i] = np.dot([X[i,:], 1], B[0].T) + e[i]*np.sqrt(sigma[0])
	else:
		Y[i] = np.dot([X[i, :], 1], B[1].T) + e[i] * np.sqrt(sigma[1])
z = pd.DataFrame(Z)
y = pd.DataFrame(Y)
x = pd.DataFrame(X)
x.insert(len(x.columns), len(x.columns), 1)

phi = np.array([[[0.5], [1]],[[0.8], [-1]]])
sigs = np.array([[[3.]], [[1.]]])
gamma = np.array([[-1],[0],[1]])
pp = 0.95 * np.ones(T)
qq = 0.95 * np.ones(T)
ncrit = 10

B0 = np.zeros((2,1))
sigma0 = np.eye(2)

d0 = 0.1
v0 = 1

GAMMA00 = np.zeros((3,1))
SGAMMA0 = np.eye(3) * 1000

u0 = np.array([25,5])
u1 = np.array([5,25])

out1 = []
out2 = []
out3 = []
out4 = []
out5 = []
out6 = []
out7 = []

reps = 20000
burn = 15000
cnt = 0
igibs = 0
while cnt < reps-burn:
	S = hamilton_filter_varyingP(y.values, x.values, pp, qq, phi, sigs, ncrit)
	S = np.where(S==1)[1]
	dfS = pd.DataFrame(S)
	# sample sstar
	sstar = np.zeros((T,1))
	slag = dfS.shift(1).fillna(0)
	slag.iloc[0,:] = slag.iloc[1,:]
	zall = np.hstack((np.ones((T,1)), z.values, slag.values))
	# gamma = loadmat('/home/wy/cslt/AppliedBayesianEconometrics/CHAPTER4/hh.mat')['gamma']
	mm = matmul(zall, gamma)
	for t in range(T):
		if S[t] == 1:
			sstar[t] = normlt_rnd(mm[t], 1, 0)
		else:
			sstar[t] = normrt_rnd(mm[t], 1, 0)
	pp = stats.norm.cdf(matmul(-zall[:,:-1],gamma[:-1,:])).flatten()
	qq = 1 - stats.norm.cdf(matmul(-zall[:,:-1],gamma[:-1,:])-gamma[-1]).flatten()

	yy = sstar
	xx = zall
	V = inv(inv(SGAMMA0) + matmul(xx.T, xx))
	M = matmul(V, matmul(inv(SGAMMA0), GAMMA00) + matmul(xx.T, yy))
	gamma = M + matmul(np.random.randn(1,3), cholesky(V).T).T

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
			out4.append(deepcopy(pp))
			out5.append(deepcopy(qq))
			out6.append(gamma.flatten())
			out7.append(sstar.flatten())
			cnt += 1

	igibs += 1
	print('replication %d, %d saved Draws' % (igibs, cnt))

B = np.squeeze(B)
plt.rc('font',size=10)
fig = plt.figure(figsize=(20,30))

plt.subplot(8,2,1)
plt.hist(np.array(out1)[:,0], 50)
plt.axvline(B[0,0], color = 'r')
plt.title('Coefficient regime 1')

plt.subplot(8,2,2)
plt.hist(np.array(out1)[:,2], 50)
plt.axvline(B[1,0], color = 'r')
plt.title('Coefficient regime 2')

plt.subplot(8,2,3)
plt.hist(np.array(out1)[:,1], 50)
plt.axvline(B[0,1], color = 'r')
plt.title('Intercept regime 1')

plt.subplot(8,2,4)
plt.hist(np.array(out1)[:,3], 50)
plt.axvline(B[1,1], color = 'r')
plt.title('Intercept regime 2')

plt.subplot(8,2,5)
plt.hist(np.array(out2)[:,0], 50)
plt.axvline(sigma[0], color = 'r')
plt.title('$\sigma_1$')

plt.subplot(8,2,6)
plt.hist(np.array(out2)[:,1], 50)
plt.axvline(sigma[1], color = 'r')
plt.title('$\sigma_2$')

plt.subplot(8,2,7)
plt.hist(np.array(out6)[:,0], 50)
plt.axvline(GAMMA[0], color = 'r')
plt.title('$\gamma_0$')

plt.subplot(8,2,8)
plt.hist(np.array(out6)[:,1], 50)
plt.axvline(LAM0, color = 'r')
plt.title('$\lambda$')

plt.subplot(8,2,9)
plt.hist(np.array(out6)[:,2], 50)
plt.axvline(GAMMA[1], color = 'r')
plt.title('$\gamma_1$')

plt.subplot(8,1,6)
tmp = np.mean(out3, axis=0)
plt.plot(tmp, 'c', linewidth=2)
plt.plot(strue, 'k', linewidth=2)
plt.title('probability of regime 1')
plt.legend(['estimate', 'true'])

plt.subplot(8,1,7)
tmp = np.mean(out4, axis=0)
plt.plot(tmp, 'c', linewidth=2)
plt.plot(ptrue, 'k', linewidth=2)
plt.title('probability of regime 1')
plt.legend(['estimate', 'true'])

plt.subplot(8,1,8)
tmp = np.mean(out5, axis=0)
plt.plot(tmp, 'c', linewidth=2)
plt.plot(qtrue, 'k', linewidth=2)
plt.title('probability of regime 1')
plt.legend(['estimate', 'true'])

plt.savefig('figures/example6.pdf')

