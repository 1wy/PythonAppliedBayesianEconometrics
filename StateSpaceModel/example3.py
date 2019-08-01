#-*- coding:utf-8 -*-
# author:wy
# datetime:18-12-27 上午9:54
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import pandas as pd
import numpy as np
from numpy import matmul, kron
from numpy.linalg import inv, cholesky
from scipy.stats import invwishart
from scipy.io import loadmat
from StateSpaceModel.utils import notstable, check_right, irfsim
from pickle import dump, load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_excel('data/usdata.xls', header=None) / 100
N = data.shape[1]
L = 2 # number of lags in the VAR

Y = data.iloc[L:,:]
X = pd.concat([data.shift(i+1) for i in range(L)], axis=1).dropna()
X.insert(len(X.columns),'constant',1)
X.columns = range(len(X.columns))
T = len(Y)

T0 = 40
Y0 = Y.iloc[:T0,:]
X0 = X.iloc[:T0,:]
b0 = matmul(inv(matmul(X0.T,X0)),matmul(X0.T, Y0))
e0 = Y0 - matmul(X0, b0)
sigma0 = matmul(e0.T, e0) / T0
V0 = kron(sigma0, inv(matmul(X0.T, X0)))

Q0 = V0 * T0 * 3.5e-4
P00 = V0
beta0 = b0.reshape(-1,1,order='F').T

Q = Q0
R = sigma0
Y = Y.iloc[T0:,:]
X = X.iloc[T0:,:]
T = len(X)

reps = 110000
burn = 109000
mm = 1
out1 = []
out2 = []
out3 = []
for m in range(reps):
	# if m % 1000 == 0:
	print("%d / %d" % (m, reps))
	ns = beta0.shape[1]
	F = np.eye(ns)
	mu = 0
	beta_tt = []
	p_tt = []
	beta11 = beta0
	p11 = P00

	for i in range(T):
		x = kron(np.eye(N), X.iloc[i,:])

		beta10 = mu + matmul(beta11, F.T)
		p10 = matmul(matmul(F, p11), F.T) + Q
		eta = Y.iloc[i,:].values - matmul(x, beta10.T).T
		feta = matmul(matmul(x, p10), x.T) + R
		# update
		K = matmul(matmul(p10, x.T), inv(feta))
		beta11 = (beta10.T + matmul(K, eta.T)).T
		p11 = p10 - matmul(K, matmul(x, p10))

		beta_tt.append(beta11)
		p_tt.append(p11)

	chck = True
	while chck:
		beta2 = np.zeros((T, ns))
		wa = np.random.randn(T, ns)
		error = np.zeros((T, N))
		roots = np.zeros((T, 1))

		i = T - 1
		p00 = p_tt[i]
		beta2[i, :] = beta_tt[i] + matmul(wa[i, :], cholesky(p00).T)
		error[i, :] = Y.iloc[i,:].values - matmul(X.iloc[i,:], beta2[i,:].reshape(-1,N,order='F'))
		roots[i] = notstable(beta2[i,:].T, N, L)

		for i in range(T-2, -1, -1):
			pt = p_tt[i]
			Kstar = matmul(matmul(pt, F.T), inv(matmul(matmul(F, pt), F.T) + Q))
			bm = beta_tt[i] + matmul(Kstar, (beta2[i + 1] - mu - matmul(beta_tt[i], F.T)).T).T
			pm = pt - matmul(matmul(Kstar, F), pt)
			beta2[i, :] = bm + matmul(wa[i, :], cholesky(pm).T)
			error[i,:] = Y.iloc[i,:].values - matmul(X.iloc[i,:], beta2[i,:].reshape(-1,N,order='F'))
			roots[i] = notstable(beta2[i,:].T, N, L)

		if np.sum(roots) == 0:
			chck = False

	errorq = np.diff(beta2, axis=0)
	scaleQ = matmul(errorq.T, errorq) + Q0
	Q = invwishart.rvs(T+T0, scaleQ)

	scaleR = matmul(error.T, error)
	R = invwishart.rvs(T, scaleR)

	if m > burn-1:
		out1.append(beta2)
		out2.append(R)
		out3.append(Q)

with open('tvp.pk','w') as f:
	dump(out1, f)
	dump(out2, f)
	dump(out3, f)

# tvp = loadmat('/home/wy/cslt/AppliedBayesianEconometrics/CHAPTER3/tvp.mat')
# out1 = tvp['out1']
# out2 = tvp['out2']
# out3 = tvp['out3']
#
# horizon = 40
# irfmat = np.zeros((len(out1), T, horizon, N))
# for i in range(len(out1)):
# 	print("%d / %d" % (i, len(out1)))
# 	sigma = out2[i]
# 	A0_hat = cholesky(sigma).T
#
# 	chck = True
# 	while chck:
# 		QQ = np.linalg.qr(np.random.randn(N, N))[0]
# 		A0 = matmul(QQ, A0_hat)
# 		symbol = np.array([-1, -1, 1])
#
# 		for m in range(N):
# 			if all(symbol * A0[m, [0, 1, 2]] > 0):
# 				MP = A0[[m], :]
# 				A0 = np.delete(A0, m, 0)
# 				A0 = np.vstack((A0,MP))
# 				chck = False
# 				break
#
# 			elif all(-1 * symbol * A0[m, [0, 1, 2]] > 0):
# 				MP = -A0[[m], :]
# 				A0 = np.delete(A0, m, 0)
# 				A0 = np.vstack((A0,MP))
# 				chck = False
# 				break
#
# 	shock = [0,0,1]
# 	for j in range(out1.shape[1]):
# 		btemp = out1[i,j,:].reshape(-1,N,order='F')
# 		zz = irfsim(btemp, N, L, A0, shock, horizon+L)
# 		zz = zz / np.tile(zz[[0],[2]],(horizon,N))
# 		irfmat[i,j,:,:] = zz
#
# TT = np.arange(1964.75,2010.75,0.25)
# HH = list(range(horizon))
#
# TT, HH = np.meshgrid(TT, HH)
# irf1 = np.median(irfmat[:,:,:,0], axis=0)
# irf2 = np.median(irfmat[:,:,:,1], axis=0)
# irf3 = np.median(irfmat[:,:,:,2], axis=0)
#
# fig = plt.figure()
#
# ax = fig.add_subplot(2,2,1,projection='3d')
# ax.plot_surface(TT, HH, irf1.T, rstride=1,cstride=1,cmap='rainbow')
# plt.title('GDP Growth')
#
# ax = fig.add_subplot(2,2,2,projection='3d')
# ax.plot_surface(TT, HH, irf2.T, rstride=1,cstride=1,cmap='rainbow')
# plt.title('Inflation')
#
# ax = fig.add_subplot(2,2,3,projection='3d')
# ax.plot_surface(TT, HH, irf3.T, rstride=1,cstride=1,cmap='rainbow')
# plt.title('Federal Funds Rate')
# plt.savefig('figures/example3.pdf')
