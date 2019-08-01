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
N = 1 # number of variable in Y
L = 1
ns = 2 # number of state
B = np.array([[0.25]])
MU = np.array([[3.], [-1]])
MUF = formatMU(MU, lag=L)
MUY = MUF[:,[0]]
MUX = MUF[:,1:]
sigma = np.array([[[1.]], [[2.]]])
sigmaF = np.tile(sigma,(len(MUF)//len(sigma),1,1))
P = np.array([[0.98, 0.02],[0.02, 0.98]])
PF = formatP(P, lag=L)
strue = np.zeros((T,4))
strue[0,0] = 1
strue = simS(strue, PF)

e = np.random.randn(T,N)
Y = np.zeros((T,N))
X = np.zeros((T,N*L))

for i in range(1,T):
	X[i,:] = Y[i-1,:]
	idx = strue[i].tolist().index(1)
	Y[i] = matmul(X[i,:]-MUY[idx], B.T) + MUX[idx] + matmul(e[i], cholesky(sigmaF[idx]).T)
# if strue[i,0] == 1:
	# 	Y[i] = matmul(X[i,:]-MU[0], B.T) + MU[0] + matmul(e[i], cholesky(sigma[0]).T)
	# elif strue[i,1] == 1:
	# 	Y[i] = matmul(X[i,:]-MU[0], B.T) + MU[1] + matmul(e[i], cholesky(sigma[1]).T)
	# elif strue[i,2] == 1:
	# 	Y[i] = matmul(X[i,:]-MU[1], B.T) + MU[0] + matmul(e[i], cholesky(sigma[0]).T)
	# else:
	# 	Y[i] = matmul(X[i,:]-MU[1], B.T) + MU[1] + matmul(e[i], cholesky(sigma[1]).T)

y = pd.DataFrame(Y)
x = pd.DataFrame(X)

phi = np.array([[[0.5]]])
phiF = np.tile(phi, (len(MUF)//len(phi),1,1))

mu = np.array([[1.], [0.1]])
muF = formatMU(mu, lag=1)
muY = muF[:,[0]]
muX = muF[:,1:]

sigs = np.array([[[1.]], [[1.]]])
sigsF = np.tile(sigs,(len(muF)//len(sigs),1,1))

p = 0.95
q = 0.95
pmat = np.array([[p, 1-q], [1-p, q]])
pmatF = formatP(pmat, lag=L)
ncrit = 10

B0 = np.array([[0.]])
sigma0 = np.eye(1)

M0 = np.zeros((2,1))
sigma0M = np.eye(2) * 10.

d0 = 0.1
v0 = 1.

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
	S = hamilton_filter_mu(y.values, x.values, pmatF, phiF, muY, muX, sigsF, ncrit)

	Sstar = 1 - S[:,::2].sum(axis=1, dtype=int)
	tranmat = switchg(Sstar, ns)
	# sample from dirichlet density
	p0 = dirichlet(tranmat[0,:] + u0, size=1).reshape(-1,1,order='F')
	p1 = dirichlet(tranmat[1,:] + u1, size=1).reshape(-1,1,order='F')
	pmat = np.hstack((p0,p1))
	pmatF = formatP(pmat, lag=L)
	# print(p0[0,0], p1[1,0])
	s = np.where(S==1)[1]

	# sample AR coefficient
	mut = np.tile(muY.T,(T,1))[range(T),s].reshape(-1,1,order='F')
	mutlag = np.array([np.tile(muX[:,[t]].T,(T,1))[range(T),s] for t in range(L)]).T
	sigall = np.tile([np.squeeze(sigsF)],(T,1))[range(T),s].reshape(-1,1,order='F')
	ystar = (y.values - mut) / np.sqrt(sigall)
	xstar = (x.values - mutlag) / np.sqrt(sigall)
	V = inv(inv(sigma0)+matmul(xstar.T, xstar))
	M = matmul(V, matmul(inv(sigma0), B0) + matmul(xstar.T, ystar))
	phi = M + matmul(np.random.randn(1,1), cholesky(V).T).T
	sigsF = np.tile(sigs, (len(muF) // len(sigs), 1, 1))

	ystar = (y.values - matmul(x, phi)) / np.sqrt(sigall)
	xstar = np.zeros((len(Sstar),ns))
	xstar[Sstar==0,0] = 1-phi
	xstar[Sstar==1,1] = 1-phi
	xstar /= sigall
	V = inv(inv(sigma0M)+matmul(xstar.T, xstar))
	M = matmul(V, matmul(inv(sigma0M), M0) + matmul(xstar.T, ystar))
	mu = M + matmul(np.random.randn(1,2), cholesky(V).T).T
	muF = formatMU(mu, lag=1)
	muY = muF[:, [0]]
	muX = muF[:, 1:]

	resid = (y.values - mut) - matmul(x.values - mutlag, phi)
	for i in range(ns):
		# sample sigma
		ei = resid[Sstar==i,:]
		scale = matmul(ei.T, ei)
		sigs[i] = [[invgamma.rvs(v0+len(ei), scale=d0 + scale)]]
	sigsF = np.tile(sigs,(len(muF)//len(sigs),1,1))
	# print(sigs[0][0,0], sigs[1][0,0])

	if igibs > burn-1:
		chck = mu[0,0] > mu[1,0] # constant bigger in regime 1
		if chck:
			out1.append(phi[0].tolist() + mu.flatten().tolist())
			out2.append(sigs.flatten())
			out3.append(Sstar)
			out4.append([p0[0,0],p1[1,0]])
			cnt += 1

	igibs += 1
	print('replication %d, %d saved Draws' % (igibs, cnt))


plt.rc('font',size=10)
fig = plt.figure(figsize=(20,20))

plt.subplot(5,2,1)
plt.hist(np.array(out1)[:,0], 50)
plt.axvline(B[0,0], color = 'r')
plt.title('Coefficient')

plt.subplot(5,2,3)
plt.hist(np.array(out1)[:,1], 50)
plt.axvline(MU[0,0], color = 'r')
plt.title('mean regime 1')

plt.subplot(5,2,4)
plt.hist(np.array(out1)[:,2], 50)
plt.axvline(MU[1,0], color = 'r')
plt.title('mean regime 2')

plt.subplot(5,2,5)
plt.hist(np.array(out2)[:,0], 50)
plt.axvline(sigma[0,0], color = 'r')
plt.title('sigma regime 1')

plt.subplot(5,2,6)
plt.hist(np.array(out2)[:,1], 50)
plt.axvline(sigma[1,0], color = 'r')
plt.title('sigma regime 2')

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
plt.plot(strue[:,1::2].sum(axis=1), 'k', linewidth=2)
plt.title('probability of regime 1')
plt.legend(['estimate', 'true'])
plt.savefig('figures/example5.pdf')
