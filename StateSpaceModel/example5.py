#-*- coding:utf-8 -*-
# author:wy
# datetime:18-12-29 上午10:46
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import pandas as pd
import numpy as np
from numpy import matmul, kron
from numpy.linalg import inv, cholesky
from scipy.stats import invwishart
from scipy.io import loadmat
from utils import *
from pickle import dump, load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nobs = 996
btrue = np.array([[0.95, 0.1],
                  [0.1, 0.95],
                  [-0.1,   0],
                  [0,   -0.1],
                  [-0.05,  0],
                  [0,  -0.05],
                  [0,      0]])
sigmatrue = np.array([[2, 1],
                      [1, 2]])

datatrue = np.zeros((nobs, 2))
noise = np.random.randn(nobs,2)
for j in range(3,nobs):
	datatrue[j,:] = matmul(np.hstack((datatrue[j-1,:], datatrue[j-2,:], datatrue[j-3,:], [1])), btrue) + \
	                matmul(noise[[j],:], cholesky(sigmatrue).T)

dataQ = np.zeros((nobs//3, 1))
jj = 0
for j in range(0,nobs,3):
	dataQ[jj,:] = datatrue[j:j+3,0].mean()
	jj += 1
dataM = datatrue[:,[1]]
dataN = np.hstack((np.nan*np.zeros((len(dataQ),2)), dataQ)).reshape(-1,1,order='C')
data0 = np.hstack((np.zeros((len(dataQ),2)), dataQ)).reshape(-1,1,order='C')

dataX = np.tile(dataQ[:,[0]],(1,3)).reshape(-1,1,order='C')
data = pd.DataFrame(np.hstack((dataX, dataM)))
dataid = np.hstack((dataN, dataM))
dataid0 = np.hstack((data0, dataM))
mid = np.isnan(dataid)

N = data.shape[1]
reps = 11000
burn = 10500

L = 3
X = pd.concat([data.shift(i+1) for i in range(L)], axis=1).dropna()
X.columns = range(X.shape[1])
X.insert(X.shape[1],X.shape[1],1)
Y = data.iloc[L:,:]
dataid0 = dataid0[L:,:]
dataM = dataM[L:,:]
T = len(X)

b0 = matmul(inv(matmul(X.T, X)), matmul(X.T, Y))
e0 = Y - matmul(X,b0)
sigma = np.eye(N)

lamP = 1
tauP = 10 * lamP
epsilonP = 1
muP = Y.mean()
sigmaP = []
deltaP = []
e0 = []

for i in range(N):
	ytemp = Y.iloc[:,[i]]
	xtemp = ytemp.shift(1).dropna()
	xtemp.insert(xtemp.shape[1],'constant',1)
	ytemp = ytemp.iloc[1:,:]
	btemp = matmul(inv(matmul(xtemp.T, xtemp)), matmul(xtemp.T, ytemp))
	etemp = ytemp - matmul(xtemp, btemp)
	stemp = matmul(etemp.T, etemp) / len(ytemp)
	if np.abs(btemp[0]) > 1:
		btemp[0] = 1
	deltaP.append(np.squeeze(btemp)[0])
	sigmaP.append(stemp[0,0])
	e0.append(etemp)

[yd, xd] = create_dummies(lamP, tauP, deltaP, epsilonP, L, muP, sigmaP, N)
beta0 = np.hstack(tuple(Y.iloc[i,:].values for i in range(L-1,-1,-1)))
P00 = np.eye(len(beta0)) * 0.1

dmat = []
bmat = []
smat = []
for gib in range(reps):
	Y0 = Y.append(pd.DataFrame(yd), ignore_index=True)
	X0 = X.append(pd.DataFrame(xd), ignore_index=True)

	mstar = matmul(inv(matmul(X0.T, X0)), matmul(X0.T, Y0)).reshape(-1,1,order='F')
	vstar = kron(sigma, inv(matmul(X0.T, X0)))
	chck = True
	while chck:
		varcoef = mstar + matmul(np.random.randn(1,N*(N*L+1)), cholesky(vstar).T).T
		if not notstable(varcoef, N, L):
			chck = False
	resids = Y0 - matmul(X0, varcoef.reshape(-1,N,order='F'))
	scaleS = matmul(resids.T, resids)
	sigma = invwishart.rvs(T, scale=scaleS)

	ns = P00.shape[1]
	F, MUx = comp(varcoef, N, L)
	Q = np.zeros((ns,ns))
	Q[:N,:N] = sigma

	beta_tt = []
	p_tt = []

	beta11 = beta0
	p11 = P00

	for i in range(T):
		nanid = mid[i,0]

		if nanid == 1:
			H = np.array([[0,0,0,0,0,0],
			              [0,1,0,0,0,0]])
			rr = np.zeros(N)
			rr[0] = 1e10
			R = np.diag(rr)
		else:
			H = np.array([[1/3, 0, 1/3, 0, 1/3, 0],
			              [0,   1, 0,   0, 0,   0]])
			rr = np.zeros(N)
			R = np.diag(rr)

		x = H

		beta10 = MUx + matmul(beta11, F.T)
		p10 = matmul(matmul(F, p11), F.T) + Q
		eta = dataid0[i,:] - matmul(x, beta10.T).T
		feta = matmul(matmul(x, p10), x.T) + R
		# update
		K = matmul(matmul(p10, x.T), inv(feta))
		beta11 = (beta10.T + matmul(K, eta.T)).T
		p11 = p10 - matmul(K, matmul(x, p10))

		beta_tt.append(beta11)
		p_tt.append(p11)
	beta_tt = np.squeeze(beta_tt)
	p_tt = np.squeeze(p_tt)

	beta2 = np.zeros((T, ns))
	bm2 = beta2
	jv = [0, 1]
	jv1 = [0,2,4]
	wa = np.random.randn(T, ns)

	f = F[jv, :]
	q = Q[jv][:, jv]
	mu = MUx[:, jv]

	i = T - 1
	p00 = p_tt[i][jv1][:, jv1]
	beta2[i,:] = beta_tt[i,:]
	beta2[i, jv1] = beta_tt[i][jv1] + matmul(wa[i, jv1], cholx(p00))

	for i in range(T - 2, -1, -1):
		pt = p_tt[i]
		Kstar = matmul(matmul(pt, f.T), inv(matmul(matmul(f, pt), f.T) + q))
		bm = beta_tt[i] + matmul(Kstar, (beta2[i + 1, jv] - mu - matmul(beta_tt[i], f.T)).T).T
		pm = pt - matmul(matmul(Kstar, f), pt)
		beta2[i, :] = bm
		beta2[i, jv1] = bm[0, jv1] + matmul(wa[i, jv1], cholx(pm[jv1][:, jv1]))
		bm2[i,:] = bm

	out = beta2[:,[0]]
	datax = pd.DataFrame(np.hstack((out, dataM)))
	Y = datax.iloc[L:, :]
	X = pd.concat([datax.shift(i + 1) for i in range(L)], axis=1).dropna()
	X.columns = range(X.shape[1])
	X.insert(X.shape[1], X.shape[1], 1)

	print('iteration number = %d' % gib)

	if gib > burn-1:
		dmat.append(np.squeeze(out))
		bmat.append(varcoef)
		smat.append(sigma)
dmat = np.array(dmat).T
fig = plt.figure(figsize=(20,10))
plt.plot(datatrue[L:,[0]],'k-',linewidth=0.8)
plt.plot(np.percentile(dmat, [50 ], 1).T, 'r--',linewidth=0.8)
plt.savefig('figures/example5.pdf')