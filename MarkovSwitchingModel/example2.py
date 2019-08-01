#-*- coding:utf-8 -*-
# author:wy
# datetime:19-1-3 下午6:59
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import numpy as np
from numpy.linalg import pinv
from numpy import matmul
from numpy.random import multinomial
from scipy.io import loadmat
import pandas as pd
from MarkovSwitchingModel.utils import simS


def hamilton_filter(Y, X, P, B, sigma, ncrit):
	"""
	estimate the hidden state with hamilton filter
	:param Y: Y in observation equation
	:param X: X in observation equation
	:param P: transition matrix
	:param B: coefficient in the observation equation, M x N, each column is the coefs of one state.
	:param sigma: variance in the observation equation
	:param ncrit: sample number restrict
	:return:
	"""
	T = len(Y)
	N = len(P)
	A = np.vstack((np.eye(N)-P, np.ones((1,N))))
	EN = np.vstack((np.zeros((N,1)), [[1]]))
	ett11 = matmul(pinv(matmul(A.T,A)), matmul(A.T,EN))
	sigma = sigma.reshape(-1,1,order='F') # N x 1
	invsigma = 1 / sigma
	lik = 0 # log likelihood
	filter = np.zeros((T,N))

	for j in range(T):
		em = np.tile(Y[j],(N,1)) - matmul(X[j,:], B).reshape(-1,1,order='F')
		neta = 1 / np.sqrt(sigma) * np.exp(-0.5*em*invsigma*em) # P(Y|S = 0)

		# prediction step
		ett10 = matmul(P, ett11)
		# update step
		ett11 = ett10 * neta
		fit = np.sum(ett11) # marginal density
		ett11 /= fit
		filter[j,:] = ett11.T
		lik += np.log(fit)

	print(lik)

	S = np.zeros((T,N))
	S[T-1,:] = multinomial(1,filter[T-1,:],size=1)

	for t in range(T-2,-1,-1):
		next_state = S[t+1].tolist().index(1)
		p = P[next_state,:] * filter[t,:]
		p = p / np.sum(p)
		S[t,:] = multinomial(1,p,size=1)

	return S

if __name__ == '__main__':

	T = 500
	B = np.array([[0.2, 1], [0.9, -1]])
	sigma = np.array([3, 1])
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
			Y[i] = np.dot([X[i,:], 1], B[0,:]) + e[i]*np.sqrt(sigma[0])
		else:
			Y[i] = np.dot([X[i, :], 1], B[1,:]) + e[i] * np.sqrt(sigma[1])

	# ============================================= Run Hamilton Filter ====================================================
	y = pd.DataFrame(Y)
	x = pd.DataFrame(X)
	x.insert(len(x.columns), len(x.columns), 1)

	S = hamilton_filter(y.values, x.values, P, B.T, sigma, 1)

	print(S)
