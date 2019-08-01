#-*- coding:utf-8 -*-
# author:wy
# datetime:19-1-3 下午8:36
# Copyright (C) Yang Wang, Tsinghua Univerity, 2018

import itertools
import numpy as np
from scipy.io import loadmat
from scipy import stats
from numpy import matmul, kron
from numpy.linalg import pinv, inv, cholesky, det
from numpy.random import multinomial

def normlt_rnd(mu, sigma, left):
	right = mu + 5*np.sqrt(sigma)
	left = (left - mu) / np.sqrt(sigma)
	right = (right - mu) / np.sqrt(sigma)
	result = stats.truncnorm.rvs(left,right,loc=mu,scale=sigma)
	return result

def normrt_rnd(mu, sigma, right):
	left = mu - 5*np.sqrt(sigma)
	left = (left - mu) / np.sqrt(sigma)
	right = (right - mu) / np.sqrt(sigma)
	result = stats.truncnorm.rvs(left,right,loc=mu,scale=sigma)
	return result

def getar(rho, T, ns):
	e = np.random.randn(T,1)
	y = np.random.randn(T,ns)
	for i in range(1,T):
		y[i,:] = np.dot(y[i-1,:],rho) + e[i]
	return y

def formatMU(MU, lag):
	ns = len(MU)
	states = [list(i) for i in itertools.product(range(ns), repeat=lag + 1)]
	newns = len(states)
	newMU = np.zeros((newns, lag+1))
	for s in range(newns):
		idx = list(states[s])
		newMU[s,:] = MU[idx,:].T
	return newMU


def formatP(P, lag):
	ns = len(P)
	states = [list(i) for i in itertools.product(range(ns), repeat=lag+1)]
	newns = len(states)
	newP = np.zeros((newns, newns))
	for s in range(newns):
		idx = [(states[s][-1]==states[s1][0]) for s1 in range(newns)]
		newP[idx,s] = P[:,states[s][-1]]
	return newP



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

def get_coef(mstar, sigma, ixx, maxtrys, N, L):
	problem = 0
	vstar = kron(sigma, ixx)

	chck = True
	tryx = 1
	while chck and (tryx<maxtrys):
		beta = mstar + matmul(np.random.randn(1,N*(N*L+1)), cholesky(vstar).T).T
		CH = notstable(beta, N, L)
		if not CH:
			chck = False
		else:
			tryx += 1
	if CH:
		problem = 1

	return beta, problem

def simS_varyingP(sin, z, gamma, lam):
	states = sin
	Sstar = np.zeros(states.shape)
	ptrue = np.zeros(Sstar.shape)
	qtrue = np.zeros(Sstar.shape)
	for t in range(1,len(sin)):
		Sstar[t,:] = gamma[0] + np.dot(z[t,:], lam) + gamma[1]*states[t-1,:] + np.random.randn(1,1)
		if Sstar[t,:] >=0:
			states[t,0] = 1
		ptrue[t] = stats.norm.cdf(-gamma[0]-np.dot(z[t,:], lam))
		qtrue[t] = 1 - stats.norm.cdf(-gamma[0]-np.dot(z[t,:], lam)-gamma[1])

	return states, ptrue, qtrue

def simS(sin, P):
	states = sin
	for t in range(1,len(sin)):
		s_t1 = states[t-1].tolist().index(1) # last state
		s_t = multinomial(1, P[:,s_t1],size=1)[0]
		states[t,:] = s_t
	return states

def hamilton_filter_mu(Y, X, P, B, muY, muX, sigma, ncrit):
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
	lik = 0 # log likelihood
	filter = np.zeros((T,N))

	for j in range(T):
		neta = np.zeros((N,1))
		for i in range(N):
			em = (Y[j]-muY[i]) - matmul(X[j,:]-muX[i], B[i])
			neta[i, 0] = 1 / np.sqrt(det(sigma[i])) * np.exp(-0.5*matmul(matmul(em,inv(sigma[i])),em.T)) # P(Y|S = 0)

		neta = np.array(neta).reshape(-1,1)
		# prediction step
		ett10 = matmul(P, ett11)
		# update step
		ett11 = ett10 * neta
		fit = np.sum(ett11) # marginal density
		ett11 /= fit
		filter[j,:] = ett11.T
		lik += np.log(fit)

	cnt = 0
	while True:
		S = np.zeros((T,N))
		S[T-1,:] = multinomial(1,filter[T-1,:],size=1)


		for t in range(T-2,-1,-1):
			next_state = S[t+1].tolist().index(1)
			p = P[next_state,:] * filter[t,:]
			p = p / np.sum(p)
			S[t,:] = multinomial(1,p,size=1)

		Scount = S.sum(axis=0)
		if all(np.array([sum(Scount[::2]), sum(Scount[1::2])])>= ncrit):
			return S
		cnt += 1
		if cnt > 100:
			pass

def hamilton_filter_varyingP(Y, X, pp, qq, B, sigma, ncrit):
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
	N = 2
	P = np.array([[pp[0], 1-qq[0]], [1-pp[0], qq[0]]])
	A = np.vstack((np.eye(N)-P, np.ones((1,N))))
	EN = np.vstack((np.zeros((N,1)), [[1]]))
	ett11 = matmul(pinv(matmul(A.T,A)), matmul(A.T,EN))
	lik = 0 # log likelihood
	filter = np.zeros((T,N))

	for t in range(T):
		P = np.array([[pp[t], 1 - qq[t]], [1 - pp[t], qq[t]]])
		neta = np.zeros((N,1))
		for i in range(N):
			em = Y[t] - matmul(X[t, :], B[i])
			neta[i, 0] = 1 / np.sqrt(det(sigma[i])) * np.exp(-0.5*matmul(matmul(em,inv(sigma[i])),em.T)) # P(Y|S = 0)

		neta = np.array(neta).reshape(-1,1)
		# prediction step
		ett10 = matmul(P, ett11)
		# update step
		ett11 = ett10 * neta
		fit = np.sum(ett11) # marginal density
		ett11 /= fit
		filter[t,:] = ett11.T
		lik += np.log(fit)

	cnt = 0
	while True:
		S = np.zeros((T,N))
		S[T-1,:] = multinomial(1,filter[T-1,:],size=1)


		for t in range(T-2,-1,-1):
			P = np.array([[pp[t+1], 1 - qq[t+1]], [1 - pp[t+1], qq[t+1]]])
			next_state = S[t+1].tolist().index(1)
			p = P[next_state,:] * filter[t,:]
			p = p / np.sum(p)
			S[t,:] = multinomial(1,p,size=1)

		if all(S.sum(axis=0) >= ncrit):
			return S
		cnt += 1
		if cnt > 100:
			pass

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
	lik = 0 # log likelihood
	filter = np.zeros((T,N))

	for t in range(T):
		neta = np.zeros((N,1))
		for i in range(N):
			em = Y[t] - matmul(X[t, :], B[i])
			neta[i, 0] = 1 / np.sqrt(det(sigma[i])) * np.exp(-0.5*matmul(matmul(em,inv(sigma[i])),em.T)) # P(Y|S = 0)

		neta = np.array(neta).reshape(-1,1)
		# prediction step
		ett10 = matmul(P, ett11)
		# update step
		ett11 = ett10 * neta
		fit = np.sum(ett11) # marginal density
		ett11 /= fit
		filter[t,:] = ett11.T
		lik += np.log(fit)

	cnt = 0
	while True:
		S = np.zeros((T,N))
		S[T-1,:] = multinomial(1,filter[T-1,:],size=1)


		for t in range(T-2,-1,-1):
			next_state = S[t+1].tolist().index(1)
			p = P[next_state,:] * filter[t,:]
			p = p / np.sum(p)
			S[t,:] = multinomial(1,p,size=1)

		if all(S.sum(axis=0) >= ncrit):
			return S
		cnt += 1
		if cnt > 100:
			pass

def switchg(s,ns):
	n = len(s)
	swt = np.zeros((ns, ns))
	for t in range(1,n):
		st1 = s[t-1]
		st = s[t]
		swt[st1,st] += 1
	return swt

def posterior_para(y, x, sig, B0, sigma0):
	V = inv(inv(sigma0) + 1/sig*matmul(x.T,x))
	M = matmul(V, matmul(inv(sigma0),B0)+(1/sig)*matmul(x.T, y))
	return M, V

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