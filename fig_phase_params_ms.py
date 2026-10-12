#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:06:26 2024

@author: soominkwon
"""

import numpy as np
import scipy.linalg as la
from data import generate_data, init_geodesic
from solver import GeodesicEstimator
from utils import projection_mse
from functools import partial
from data_obs import generate_ms
from altmin import fit_altmin
import matplotlib.pyplot as plt


def spectral_init(Ys, As):
    """
    Performing spectral initialization for matrix sensing.
    """

    T = len(Ys)
    ell = Ys[0].shape[1]
    
    Xs_init = []

    for t in range(T):
        temp = []
        for i in range(ell):
            spect_mat = np.diag( Ys[t].squeeze() ) @ As[t]
            temp.append(np.expand_dims( np.linalg.svd( spect_mat.T )[0][:, 0], axis=1 ))

        Xs_init.append(np.concatenate(temp, axis=1))
        
    return Xs_init



d = 80
ell = 1
k = 3

Ts = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110]
# FRAC_OBSERVED = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
start = 0.25
end = 0.75
step = 0.025
FRAC_OBSERVED = np.arange(start, end + step, step).tolist()
ms = [int(frac*d) for frac in FRAC_OBSERVED]


SIGMA = 0.0
NTRIALS = 10
NITER = 10_000

# For AltMin
TOL = 1e-3
N_INNER_ITERS = 10



phase_mat = np.zeros((len(Ts), len(FRAC_OBSERVED)))

for i in range(len(Ts)):
    T = Ts[i]
    print("T:", T)
    for j in range(len(FRAC_OBSERVED)):
        m = ms[j]

        for trial in range(NTRIALS):

            # Init
            err_U_mm = np.zeros((NITER, ))
            
            # Generate data
            geo_true, Xs, ts, Gs = generate_data(d=d, k=k, ell=ell, T=T)
            forward, transpose, masks = generate_ms(d=d, ell=ell, T=T, m=m)

            # Generate masked data
            Ys = []

            for p, Xi in enumerate(Xs):
                Ys.append( forward(Xi, masks[p]) )

            # Initialize data
            Xs_init = spectral_init(Ys=Ys, As=masks)
            geo_init = init_geodesic(d=d, k=k, Xs=Xs_init, init="pca") # Initialize geodesic

            # For Geodesic
            def callback(itr, geo, data):

                # Computing subspace error
                for _, ti in enumerate(ts):
                    err_U_mm[itr] += projection_mse( geo(ti), geo_true(ti) )
                err_U_mm[itr] = err_U_mm[itr] / T

                # Early stopping for success
                if err_U_mm[itr] < 0.0099:
                    
                    phase_mat[i, j] += 1
                    print("Trial:", trial, "Success")
                    
                    return True
                
                # Early stopping for failure
                if itr > 1000:
                     if err_U_mm[itr-1] < err_U_mm[itr]: # Stop if error is increasing
                            print("Trial:", trial, "Failure")
                            
                            return True

                return False

            data_init = {
                "Xs":   Xs_init,
                "ts":   ts,
                "measurements":  Ys,
                "forward":   forward,
                "transpose": transpose,
                "masks": masks,

            }
            estimator = GeodesicEstimator(rank=k)
            geo_mm_est = estimator.fit(
                            geo=geo_init,
                            data=data_init,
                            callback=callback,
                            niters=NITER,  
            )
            
np.savez('phase_matrix_d80_ms.npz', phase_mat)