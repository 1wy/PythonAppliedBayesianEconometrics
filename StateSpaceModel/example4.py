#-*- coding:utf-8 -*-
# author:wy
# datetime:18-12-27 上午9:54
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import pandas as pd
import numpy as np
from numpy import matmul, kron
from numpy.linalg import inv
from scipy.stats import invwishart, invgamma
from scipy.io import loadmat
from utils import *
from pickle import dump, load
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ================================================= read and preprocess data ===========================================
data0 = pd.read_excel('data/datain.xls', header=None)
junk = pd.read_excel('data/names.xls', header=None)
index = pd.read_excel('data/index.xls', header=None)

dindex = index.iloc[:,0] # dindex=1 for series that are log differenced dindex=3 differencing without logs
index = index.iloc[:,1] # index=1 for 'fast moving' series

data = []
for i in range(data0.shape[1]):
	if dindex[i] == 1:
		dat = np.diff(np.log(data0.iloc[:,i].values)) * 100
	elif dindex[i] == 3:
		dat = np.diff(data0.iloc[:,i].values)
	else:
		dat = data0.iloc[1:,i].values
	data.append(dat)

data = pd.DataFrame(np.transpose(data))
data = (data - data.mean()) / data.std()

z = pd.read_excel('data/baserate.xls', header=None).iloc[1:,:]
z = (z - z.mean()) / z.std()
# ======================================================================================================================

KK = 3 # number of factors
L = 2 # lag
N = KK + 1 # factors plus the interest rate
NN = data.shape[1]
T = len(data)

pmat, _ = pca(data.values, KK)
beta0 = np.hstack((pmat[[0],:], z.iloc[[0],:].values, np.zeros((1,N))))
ns = beta0.shape[1]
P00 = np.eye(ns)
rmat = np.ones((NN,1))
sigma = np.eye(N)

reps = 5000
burn = 4000
mm = 1
irfmat = []
for m in range(reps):
	print("%d / %d" % (m, reps))
	# step 1
	fload = []
	floadr = []
	error = []
	for i in range(NN):
		y = data.iloc[:,[i]]
		if index[i] == 0:
			x = pmat
		else:
			x = np.hstack((pmat, z))

		M = matmul(inv(matmul(x.T, x)), matmul(x.T, y))
		V = rmat[i] * inv(matmul(x.T, x))

		ff = M + matmul(np.random.randn(1, x.shape[1]), cholx(V)).T
		if index[i] == 0:
			fload.append(ff.T)
			floadr.append([0])
		else:
			fload.append(ff[:-1].T)
			floadr.append([ff[-1]])

		error.append((y.values-matmul(x, ff)).T)

	error = np.squeeze(error).T
	fload = np.squeeze(fload)
	floadr = np.array(floadr)
	fload[:KK, :KK] = np.eye(KK)
	floadr[23:23+KK,:] = np.zeros((KK,1))

	# step 2
	rmat = [invgamma.rvs(len(error), scale=np.dot(error[:,i], error[:,i])) for i in range(NN)]
	Y = pd.DataFrame(np.hstack((pmat, z.values)))
	X = pd.concat([Y.shift(i + 1) for i in range(L)], axis=1).fillna(0)
	X.insert(len(X.columns), 'constant', 1)
	Y = Y.iloc[1:,:]
	X = X.iloc[1:,:]

	M = matmul(inv(matmul(X.T, X)), matmul(X.T, Y)).reshape(-1,1,order='F')
	V = kron(sigma, inv(matmul(X.T, X)))

	chck = True
	while chck:
		beta = M + matmul(np.random.randn(1,N*(N*L+1)), cholx(V)).T
		if not notstable(beta, N, L):
			chck = False

	beta1 = beta.reshape(-1,N,order='F')
	errorsv = Y - matmul(X, beta1)
	scale = matmul(errorsv.T, errorsv)
	sigma = invwishart.rvs(T, scale=scale)

	# prepare matrices for state space
	H = np.zeros((NN+1, (KK+1)*L))
	H[:len(fload), :KK+1] = np.hstack((fload, floadr))
	H[len(floadr), KK] = 1

	R = np.diag(np.append(rmat, [0]))
	MU = np.vstack((beta1[[-1],:].T, np.zeros((N*(L-1), 1)))).T
	F = np.vstack((beta1[:N*L, :].T, np.eye(N*(L-1),N*L)))
	Q = np.zeros(F.shape)
	Q[:N,:N] = sigma

	# Carter and Kohn algorithm to draw the factor
	beta_tt = []
	p_tt = []

	x = H
	i = 0
	beta10 = MU + matmul(beta0, F.T)
	p10 = matmul(matmul(F, P00), F.T) + Q
	eta = np.hstack((data.iloc[[i],:].values, z.iloc[[i],:].values)) - matmul(x, beta10.T).T
	feta = matmul(matmul(x, p10), x.T) + R
	# update
	K = matmul(matmul(p10, x.T), inv(feta))
	beta11 = (beta10.T + matmul(K, eta.T)).T
	p11 = p10 - matmul(K, matmul(x, p10))

	beta_tt.append(beta11)
	p_tt.append(p11)

	for i in range(1,T):
		# prediction
		beta10 = MU + matmul(beta11, F.T)
		p10 = matmul(matmul(F, p11), F.T) + Q
		eta = np.hstack((data.iloc[[i],:].values, z.iloc[[i],:].values)) - matmul(x, beta10.T).T
		feta = matmul(matmul(x, p10), x.T) + R
		# update
		K = matmul(matmul(p10, x.T), inv(feta))
		beta11 = (beta10.T + matmul(K, eta.T)).T
		p11 = p10 - matmul(K, matmul(x, p10))

		beta_tt.append(beta11)
		p_tt.append(p11)

	beta2 = np.zeros((T, ns))
	jv = list(range(3))
	wa = np.random.randn(T, ns)

	f = F[jv,:]
	q = Q[jv][:, jv]
	mu = MU[:,jv]

	i = T - 1
	p00 = p_tt[i][jv][:,jv]
	beta2[i, jv] = beta_tt[i][0,jv] + matmul(wa[i, jv], cholx(p00))

	for i in range(T - 2, -1, -1):
		pt = p_tt[i]
		Kstar = matmul(matmul(pt, f.T), inv(matmul(matmul(f, pt), f.T) + q))
		bm = beta_tt[i] + matmul(Kstar, (beta2[i+1,jv] - mu - matmul(beta_tt[i], f.T)).T).T
		pm = pt - matmul(matmul(Kstar, f), pt)
		beta2[i, jv] = bm[0,jv] + matmul(wa[i, jv], cholx(pm[jv][:,jv]))

	pmat = beta2[:,:3]
	if m > burn-1:
		A0 = cholx(sigma)
		y_hat = np.zeros((36,N))
		v_hat = np.zeros((36,N))
		v_hat[2,:N] = [0,0,0,1]

		for i in range(2,36):
			y_hat[i,:] = matmul(np.hstack((y_hat[i-1,:], y_hat[i-2,:], [1])),
			                    np.vstack((beta1[:N*L,:], np.zeros((1,N))))) + matmul(v_hat[i,:], A0)

		y_hat1 = matmul(y_hat, H[:,:KK+1].T)
		irfmat.append(y_hat1)

irf = np.percentile(irfmat, [50,16,84],axis=0)
junk = junk.append({0:'Interest'},ignore_index=True)
fig = plt.figure()
j = 1
for i in range(irf.shape[2]):
	plt.subplot(5,10,j)
	plt.plot(irf[:,2:,i].T)
	plt.title(junk.iloc[i,0], fontsize=5)
	j += 1

plt.savefig('figures/example4.pdf')