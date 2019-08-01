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
N = 2 # number of variable in Y
L = 1
ns = 2 # number of state
B = np.array([[[0.2, -0.1, -1], [0.5, -0.1, -1]],
              [[0.5,  0.1,  1], [0.7,  0.1,  1]]])
sigma = np.array([[[3, -0.5], [-0.5, 3]],
                  [[1,  0.1], [ 0.1, 1]]])
P = np.array([[0.95, 0.05],[0.05, 0.95]])
strue = np.zeros((T,2))
strue[0,0] = 1
strue = simS(strue, P)

e = np.random.randn(T,N)
Y = np.zeros((T,N))
X = np.zeros((T,N*L+1))

for i in range(1,T):
	X[i,:] = np.append(Y[i-1,:], [1])
	if strue[i,0] == 1:
		Y[i] = matmul(X[i,:], B[0].T) + matmul(e[i], cholesky(sigma[0]).T)
	else:
		Y[i] = matmul(X[i,:], B[1].T) + matmul(e[i], cholesky(sigma[1]).T)
y = pd.DataFrame(Y)
x = pd.DataFrame(X)

maxtrys = 1000
phiols = matmul(inv(matmul(x.T,x)), matmul(x.T,y))
phi = [deepcopy(phiols) for _ in range(ns)]
phi_last = deepcopy(phi)
sigs = [3.*np.eye(N), np.eye(N)]
p = 0.95
q = 0.95
pmat = np.array([[p, 1-q], [1-p, q]])
ncrit = 10

# VAR coefficients and variance priors via dummy observations
lamP = 10
tauP = 10*lamP
epsilonP = 1/10000
muP = Y.mean(axis=0)
sigmaP = []
deltaP = []
e0 = []
for i in range(N):
	y_temp = y.iloc[:,[i]]
	x_temp = y_temp.shift(1)
	x_temp.insert(len(x_temp.columns),'constant',1)
	y_temp = y_temp.iloc[1:,:]
	x_temp = x_temp.iloc[1:,:]

	b_temp = matmul(inv(matmul(x_temp.T, x_temp)), matmul(x_temp.T,y_temp))
	e_temp = y_temp - matmul(x_temp,b_temp)
	s_temp = matmul(e_temp.T, e_temp) / len(y_temp)
	if np.abs(b_temp[0] > 1):
		b_temp[0] = 1
	deltaP.append(np.squeeze(b_temp)[0])
	sigmaP.append(s_temp[0,0])
	e0.append(e_temp.values.flatten(order='F'))

yd, xd = create_dummies(lamP, tauP, deltaP, epsilonP, L, muP, sigmaP, N)
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
	tranmat = switchg(S)
	S = np.where(S == 1)[1]
	# sample from dirichlet density
	p0 = dirichlet(tranmat[0,:] + u0, size=1).reshape(-1,1,order='F')
	p1 = dirichlet(tranmat[1,:] + u1, size=1).reshape(-1,1,order='F')
	pmat = np.hstack((p0,p1))
	# print(p0[0,0], p1[1,0])

	for i in range(N):
		yi = y[S == i].values
		xi = x[S == i].values
		Y0 = np.vstack((yi, yd))
		X0 = np.vstack((xi, xd))

		# sample beta
		M = matmul(inv(matmul(X0.T, X0)), matmul(X0.T, Y0)).reshape(-1,1,order='F')
		ixx = inv(matmul(X0.T, X0))
		phitmp, problem = get_coef(M, sigs[i], ixx, maxtrys, N, L)
		if problem:
			phitmp = phi_last[i]
		else:
			phi_last[i] = phitmp

		phi[i] = phitmp.reshape(-1,N,order='F')

		# sample sigma
		ei = Y0 - matmul(X0, phi[i])
		scale = matmul(ei.T, ei)
		sigs[i] = invwishart.rvs(len(Y0), scale=scale)
	# print(sigs[0][0,0], sigs[1][0,0])

	if igibs > burn-1:
		chck = det(sigs[0]) > det(sigs[1]) # constant bigger in regime 1
		if chck:
			tmp = np.array([n.flatten('F') for n in phi])
			out1.append(tmp.flatten('C'))
			out2.append(deepcopy(sigs))
			out3.append(S)
			out4.append([p0[0,0],p1[1,0]])
			cnt += 1

	igibs += 1
	print('replication %d, %d saved Draws' % (igibs, cnt))

plt.rc('font',size=10)
fig = plt.figure(figsize=(20,25))

trueB = B.flatten(order='C')
for i in range(len(trueB)):
	plt.subplot(7,2,i+1)
	plt.hist(np.array(out1)[:, i], 50)
	plt.axvline(trueB[i], color='r')
	plt.title('coef %d' % (i+1))

plt.subplot(7,1,7)
tmp = np.mean(out3, axis=0)
plt.plot(tmp, 'c', linewidth=2)
plt.plot(strue[:,1], 'k', linewidth=2)
plt.title('probability of regime 1')
plt.legend(['estimate', 'true'])
plt.savefig('figures/example4.pdf')

