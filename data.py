#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 22:50:57 2023

@author: soominkwon
"""

import numpy as np
import core as _core
import utils as _utils
import scipy.linalg as la

class GrassmannGeodesic(object):
    pass


class TwoPointGeodesic(GrassmannGeodesic):
    """Docstring for TwoPointGeodesic"""

    def __init__(self, A, B, tA=0.0, tB=1.0):
        super(TwoPointGeodesic, self).__init__()
        self.A = A
        self.B = B
        self.tA = tA
        self.tB = tB

        assert A.shape == B.shape
        assert A.ndim == 2
        assert _utils.check_ortho(A)
        assert _utils.check_ortho(B)
        assert tA != tB

    def __call__(self, t):
        Y, theta, Z = _core.svd_log_map(self.A, self.B)
        if hasattr(t, "ndim"):
            t = t[..., None, None]
        arg = theta * (t - self.tA) / (self.tB - self.tA)
        return self.A @ (Z * np.cos(arg)) + Y * np.sin(arg)

    def as_point_tangent(self, tH=0.0):
        H0 = _core.geodesic_eqn(self.A, *_core.svd_log_map(self.A, self.B), t=tH)
        Y, theta0, Z = _core.svd_log_map(H0, self.B)
        H = H0 @ Z
        theta = theta0 / (self.tB - tH)
        return PointTangentGeodesic(H, Y, theta, tH)


class PointTangentGeodesic(GrassmannGeodesic):
    """Docstring for PointTangentGeodesic"""

    def __init__(self, H, Y, theta, tH=0.0):
        super(PointTangentGeodesic, self).__init__()
        self.H = H
        self.Y = Y
        self.theta = np.atleast_1d(theta)
        self.tH = tH

        assert H.shape == Y.shape
        assert H.ndim == 2
        assert self.theta.ndim == 1
        assert H.shape[1] == self.theta.shape[0]
        assert _utils.check_ortho(H)
        assert _utils.check_ortho(Y)
        assert _utils.check_ortho(H, Y)

    def __call__(self, t):
        if hasattr(t, "ndim"):
            t = t[..., None, None]
        arg = (t - self.tH) * self.theta
        return self.H * np.cos(arg) + self.Y * np.sin(arg)

    def as_two_point(self, tA=0.0, tB=1.0):
        return TwoPointGeodesic(self(tA), self(tB), tA, tB)

    def standardize(self):
        self.Y *= np.sign(self.theta)
        self.theta = np.abs(self.theta)
        ii = np.argsort(self.theta)[::-1]
        self.H = self.H[:, ii]
        self.Y = self.Y[:, ii]
        self.theta = self.theta[ii]
        
    
def init_geodesic(d, k, A=None, B=None, Xs=None, init="random"):
    """
    Initialize geodesic parameters.
        
    Arguments:
        d:    Dimensions of signal
        k:    Rank of subspace
        X:    Data to perform PCA-based initialization if available
        init: Type of initialization
    """
    
    init = init.lower()
    if init in ["random", "rand"]:
        
        if A is None and B is None:
            A = _core.random_subspace((d, k))
            B = _core.random_subspace((d, k))
            
    elif init in ["svd", "pca"]:
        if Xs is None:
            raise Exception("No data has been given for PCA initialization!")
        else:
            if isinstance(Xs, list):
                X_full = np.concatenate(Xs, axis=1)
            else:
                X_full = Xs
                
            A = la.svd(X_full[:, : 1 * k])[0][:, :k]
            B = la.svd(X_full[:, -1 * k :])[0][:, :k]

    else:
        raise ValueError(f"Init string '{init}' not recognized")
    geo = TwoPointGeodesic(A, B).as_point_tangent()

    return geo


def generate_data(d, k, ell, T):
    """
    Function to generate a tuple of data.
    """

    ts = np.linspace(0.0, 1.0, T) # Timestamps
    geo = init_geodesic(d=d, k=k) # Initializing geodesic parameters
    Gs = [np.random.randn(k, ell) for i in range(T)] # Generating loading vectors
    Xs = [geo(ts[i]) @ Gs[i] for i in range(T)]    

    return (geo, Xs, ts, Gs)


