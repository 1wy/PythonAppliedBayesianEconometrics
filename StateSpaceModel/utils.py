#-*- coding:utf-8 -*-
# author:wy
# datetime:18-12-19 下午8:46
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import numpy as np
from numpy import matmul
from numpy.linalg import pinv, inv, eig, cholesky
from scipy.io import loadmat
from scipy.linalg import sqrtm

def comp(beta, n, l):
	FF = np.zeros((n * l, n * l))
	FF[n:n * l, :n * (l - 1)] = np.eye(n * (l - 1))

	beta = beta.reshape(-1, n, order='F')
	temp = beta[:n*l, :].T
	FF[:n, :n*l] = temp
	temp = beta.reshape(-1,n,order='F')
	mu = np.zeros((n*l, 1))
	mu[:n] = temp[[-1],:].T
	mu = mu.T
	return FF, mu

def cholx(x):
	try:
		A = cholesky(x).T
	except:
		A = np.real(sqrtm(x)).T
	return A

def pca(data, k):
	# extract first k principal components from maxtrix(TxN)
	# and return the factors(Txk) as well as the normalized
	# principal components lam(Nxk)
	ncol = data.shape[1]
	xx = matmul(data.T, data)
	w, v = eig(xx)
	eig_pairs = [(np.abs(w[i]), v[:,i]) for i in range(len(w))]
	eig_pairs.sort(key=lambda x:x[0], reverse=True)
	v = np.array([eig_pairs[i][1] for i in range(k)]).T

	lam = v * np.sqrt(ncol)
	fac = matmul(data, lam) / ncol
	return fac, lam
# def mlikvar1(y, x, yd, xd):
# 	# not finished yet
# 	N = y.shape[1]
# 	T = y.shape[0]
# 	v = len(yd)
#
# 	y1 = np.vstack((y,yd))
# 	x1 = np.vstack((x,xd))
#
# 	xx0 = matmul(xd.T, xd)
# 	invxx0 = pinv(xx0)
# 	b0 = matmul(invxx0, matmul(xd.T, yd))
# 	v0 = len(yd)
# 	e0 = yd - matmul(xd, b0)
# 	sigma0 = matmul(e0.T, e0)
#
# 	xx1 = matmul(x1.T, x1)
# 	invxx1 = pinv(xx1)
# 	b = matmul(invxx1, matmul(x1.T, y1))
# 	v1 = v0 + T
# 	e = y1 - matmul(x1, b)
# 	sigma1 = matmul(e.T, e)
#
# 	PP = inv(np.eye(T) + matmul(matmul(x,invxx0),x.T))
# 	QQ = sigma0


def irfsim(b, n, l, v, s, t):
	"""
	:param b: VAR coefs
	:param n: number of variables
	:param l: lag length
	:param v: v = A0 matrix
	:param s: shock vector
	:param t: horizon
	:return y:
	"""
	e = np.zeros((t+l,n))
	e[l,:] = s
	y =np.zeros((t+l,n))
	for k in range(l,t):
		x = []
		for i in range(l):
			for j in range(n):
				x.append(y[k-i-1,j])
		y[k,:] = np.matmul(np.hstack((x,[0])), b) + np.matmul(e[k,:],v)

	y = y[l:len(y)-l,:]
	return y

def check_right(data, groundTruthDir, key):
	groundTruth = loadmat(groundTruthDir)[key]
	data = np.squeeze(data)
	groundTruth = np.squeeze(groundTruth)
	return np.sum(np.abs(data - groundTruth)) < 1e-6

def notstable(beta, n, l):
	FF = np.zeros((n*l,n*l))
	FF[n:n*l,:n*(l-1)] = np.eye(n*(l-1))

	beta = beta.reshape(-1, n, order='F')
	temp = beta[:-1,:].T
	FF[:n,:n*l] = temp
	ee = np.max(np.abs(np.linalg.eigvals(FF)))
	return ee > 1

def create_dummies(lam, tau, delta, epsilon, L, mu, sigma, N):
	delta = np.array(delta)
	mu = np.array(mu)
	sigma = np.array(sigma)
	if lam > 0:
		if epsilon > 0:
			yd1 = np.vstack((np.diag(sigma*delta)/lam,
			                 np.zeros((N*(L-1),N)),
			                 np.diag(sigma),
			                 np.zeros((1,N))))
			jp = np.diag(range(1,L+1))
			xd1 = np.vstack((np.hstack((np.kron(jp, np.diag(sigma)/lam), np.zeros((N*L,1)))),
			                 np.zeros((N,N*L+1)),
			                 np.hstack((np.zeros((1,N*L)), [[epsilon]]))))
		else:
			yd1 = np.vstack((np.diag(sigma*delta)/lam,
			                 np.zeros((N*(L-1),N)),
			                 np.diag(sigma)))
			jp = np.diag(range(1,L+1))
			xd1 = np.vstack((np.kron(jp, np.diag(sigma)) / lam,
			                 np.zeros((N,N*L))))
	else:
		raise("value error")

	if tau > 0:
		if epsilon > 0:
			yd2 = np.diag(delta*mu) / tau
			xd2 = np.hstack((np.kron(np.ones((1,L)), yd2), np.zeros((N,1))))
		else:
			yd2 = np.diag(delta*mu) / tau
			xd2 = np.kron(np.ones((1,L)), yd2)
	else:
		raise("value error")

	y = np.vstack((yd1, yd2))
	x = np.vstack((xd1, xd2))
	return (y, x)