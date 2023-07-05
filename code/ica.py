#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 17:42:11 2022

@author: bijoux
"""

import numpy as np
import scipy.signal
import math

from numpy import array, linspace, apply_along_axis
from numpy import dot, diag
from numpy import real, sqrt, exp, pi, tanh, sin, cos, log10
from numpy.linalg import eig, inv
from numpy.random import rand, randn

mh = lambda v: v.reshape(1, len(v))
mv = lambda v: v.reshape(len(v), 1)

# wireless toolkit

prbs = lambda length: (rand(length) < .5).astype('bool') # pseudo random binary stream (basic)
noise = lambda shape: crandn(*shape) # gaussian noise

def bpsk(b=None, length=False):
    if length:
        return 1 # return bit length
    im = 0
    re = 1 if b[0] else -1
    return re + 1j * im

def qpsk(b=None, length=False):
    if length:
        return 2 # return bit length
    re = 1 if b[0] else -1
    im = 1 if b[1] else -1
    return re + 1j * im

def qam16(b=None, length=False):
    if length:
        return 4 # return bit length
    if b[0]:
        re = 0.5 if b[1] else 1.5
    else:
        re = -0.5 if b[1] else -1.5
    if not b[2] and not b[3]:
        im = .5
    elif not b[2] and b[3]:
        im = -.5
    elif b[2] and not b[3]:
        im = 1.5
    else:
        im = -1.5
    return (re + 1j * im) / sqrt(2.5) # 正規化定数が 2.5 でいいか要確認

def encode(stream, func):
    c = func(length=True)
    b = stream.reshape(c, int(len(stream) / c))
    return apply_along_axis(func, 0, b)

# statistic toolkit

crandn = lambda *shape: (randn(*shape) + 1j * randn(*shape)) / sqrt(2)
# normalize matrix in a direction of column
normv = lambda w: dot(w,diag(diag(dot(w.T.conj(),w))**(-.5)))
# normalize matrix in a direction of row
normh = lambda w: dot(diag(diag(dot(w.T.conj(),w))**(-.5)),w)

def cvar(x):
    m = x.mean(1)
    xo = x - array([m]).T
    v  = diag(real(dot(xo,xo.T.conj()))) / x.shape[1]
    return v, m

def orth(w):
    d, v = eig(dot(w,w.T.conj()))
    w = dot(dot(dot(v,diag(d**(-.5))),v.T.conj()),w)
    return w

# uniform distribution complexed random values
def crand(m,n):
    o = 200 if n < 100 else 2 * n
    r = 2 * rand(m,o) - 1
    i = 2 * rand(m,o) - 1
    b = sqrt(r * r + i * i) < 1
    r = array([ri[bi.nonzero()][:n] for ri,bi in zip(r,b)])
    i = array([ii[bi.nonzero()][:n] for ii,bi in zip(i,b)])
    return r + 1j * i
    
def norm_mv(x):
    v , m  = cvar(x)
    vt, mt = array([v]).T, array([m]).T
    return (x - mt) / sqrt(vt), m, v
    
# ica toolkit

def ica(z, func, w=None, max_iter=100, eps=1e-12):
    def delta(w1,w2):
        n, neg, pos = w1.shape[0], w1 - w2, w1 + w2
        dn  = diag(real(dot(neg.T.conj(),neg))).sum() / n
        dp  = diag(real(dot(pos.T.conj(),pos))).sum() / n
        return min(dn,dp)
    m, n = z.shape
    w = orth(normv(crand(m,m))) if w == None else w
    w0 = w.copy()
    for i in range(max_iter):
        wo = w.copy()
        w = func(z, w)
        err = delta(w, wo)
        if err < eps:
            success = True
            break
    else:
        success = False
    params = {
        'iter': i,
        'success': success,
        'err': err,
        'max_iter': max_iter,
        'eps': eps,
        'w0': w0,
        }
    return w.T.conj(), params

def cfica(z, w):
    m, n = z.shape
    for p in range(m):
        wp = w[:, p]
        wph = wp.conj()
        wz  = dot(wph,z)
        wz2 = wz * wz.conj()
        g   = tanh(wz2)
        gp  = 1 - g ** 2
        w1  = z * wz.conj() * g
        w2  = g + wz2 * gp
        wn  = w1.mean(1) - w2.mean() * wp
        w[:, p] = dot(wn,wn)**(-.5) * wn
    return orth(w)


def white(x,n=None):
    if n == None:
      n = x.shape[0]
    si = x.shape[0] - n
    c    = dot(x,x.conj().transpose()) / x.shape[1]
    d0, e = eig(c)
    idx  = d0.argsort()
    d, e = d0[idx[si:]], e[:,idx[si:]]
    u    = dot(diag(d**(-.5)),e.conj().transpose())
    z    = dot(u,x)
    return z, u, d0[idx]


# antenna toolkit

# # direction vector
# def direction_vector(t):
#     DEG = pi / 180
#     return array([sin(t*DEG),cos(t*DEG),0*t]) # direction vector


# def beamformer(rxx, p, ph = linspace(-90,90,191)):
#     b = exp(2j*pi*dot(p,direction_vector(ph)))
#     y2 = (dot(b.T.conj(),rxx).T * b).sum(axis=0) / (b.conj() * b).sum(axis=0)
#     return ph, real(y2)

def permutation(y, s):
    ya, m, sgm = norm_mv(y)
    sa, m, sgm = norm_mv(s)
    c = dot(sa,ya.T.conj()) / sa.shape[-1] # 共分散行列
    a = abs(c)
    b = ((a / array([a.max(axis=-1)]).T) == 1).astype(int) # 並替行列 Permutation matrix
    # bc = b * c / a
    # ya = dot(b,y)
    # yb = dot(bc,y)
    yb = dot(b * c / abs(c), y)
    sgm = abs(dot(b,c)).max(axis=-1).reshape(len(c),1)
    return yb, b, dot(b,c), sgm
    
def decompose(wh, x):
    w = wh.T.conj()
    whmp = dot(w, inv(dot(wh, w)))
    wh = [dot(mv(whmp[:,i]),mh(wh[i,:])) for i in range(wh.shape[0])]
    xdcmp = [dot(whi, x) for whi in wh]
    return wh, xdcmp

# s, x, wh, p, t

def calc_sinr(yb, s):
    c = dot(s,yb.T.conj()) / s.shape[-1]
    sgm = abs(c).max(axis=-1).reshape(len(c),1)
    sinr = 10*log10(real((yb/sgm-s) * (yb/sgm-s).conj()).sum(axis=-1) / s.shape[-1])
    return sinr

# def calc_ffp_beamformer(x, a):
#     rxx = dot(x,x.T.conj()) / x.shape[-1]
#     y2 = (dot(a.T.conj(),rxx).T * a).sum(axis=0) / (a.conj() * a).sum(axis=0)
#     return real(y2)

# def calc_ffp(x, wh, ph=linspace(-90,90,181)):
#     N = x.shape[0]
#     ffp = []
#     for xi in x:
#         # rxx = dot(xi,xi.T.conj()) / xi.shape[-1]
#         # ph, y2 = calc_ffp_beamformer(rxx,p)
#         a = calc_array_manifold_lineararray(ph, N)
#         y2 = calc_ffp_beamformer(x, a)
#         ffp.append([ph, y2])
#     ffp = array(ffp)


def capon2(rxx, am):
    def func(i):
        ami = am[:, i].reshape(am.shape[0], 1)
        amh = ami.T.conj()
        spci = np.abs(1. / np.dot(np.dot(amh, np.linalg.inv(rxx)), ami))
        return spci[0, 0]
    return np.array([func(i) for i in range(am.shape[-1])])

def capon(x, am):
    rxx = np.dot(x, x.T.conj()) / x.shape[-1]
    return capon2(rxx, am)

def music(x, am, n):
    m = n - x.shape[0]
    if m < 0:
        rxx = np.dot(x, x.T.conj()) / x.shape[-1]
        d, e = np.linalg.eig(rxx)
        e = e[:, n:]
        eh = e.T.conj()
        spc = []
        for i in range(am.shape[1]):
            a = am[:,i].reshape(am.shape[0],1)
            ah = a.T.conj()
            spci = np.dot(ah, a) / np.dot(np.dot(np.dot(ah, e), eh), a)
            spc.append(spci[0,0].conj() * spci[0,0])
        return np.abs(np.array(spc))
    else:
        return np.ones(am.shape[1])


def beamformer(rxx, a):
    y2 = (dot(a.T.conj(),rxx).T * a).sum(axis=0) / (a.conj() * a).sum(axis=0)
    return real(y2)

# def find_argpeaks(spc, n):
#     idx = scipy.signal.argrelmax(spc)[0]
#     val = np.array([spc[i] for i in idx])
#     return idx[val.argsort()[:-n-1:-1]]

# def find_argpeak(func, xmin, xmid, xmax, cmax = 10, count=0):
#     rslt = xmid
#     ymin, ymid, ymax = func(xmin), func(xmid), func(xmax)
#     if ymin > ymid or ymax > ymid:
#         raise(ValueError, 'peak must be in open section (xmin:xmax).')
#     pmin, pmid, pmax =.5*(xmin+xmid), xmid, .5*(xmid+xmax)
#     qmin, qmid, qmax = func(pmin), ymid, func(pmax)
#     if count > cmax:
#         dgt = math.log10(pmax - pmin)
#         sig_dgt = -int((1 if dgt > 0 else -1) * (abs(dgt) - 1))
#         return round(pmid, sig_dgt) # , sig_dgt, count
#     if qmid < qmin and qmid < qmax:
#         raise(ValueError, 'two or more peaks detected in (xmin:xmax).')
#     elif qmid < qmin and qmid > qmax: # search in min side
#         rslt = find_argpeak(func, xmin, pmin, pmid, cmax = cmax, count=count+1)
#     elif qmid > qmin and qmid < qmax: # search in max side
#         rslt = find_argpeak(func, pmid, pmax, xmax, cmax = cmax, count=count+1)
#     elif qmid > qmin and qmid > qmax:
#         rslt = find_argpeak(func, pmin, pmid, pmax, cmax = cmax, count=count+1)
#     return rslt
