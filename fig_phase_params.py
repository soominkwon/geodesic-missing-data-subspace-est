#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 01:28:06 2024

@author: soominkwon
"""

import numpy as np
import scipy.linalg as la
from data import generate_data, init_geodesic
from solver import GeodesicEstimator
from utils import projection_mse
from functools import partial
from data_obs import generate_mc
from altmin import fit_altmin
import matplotlib.pyplot as plt


def spectral_init(Ys, frac_observed):
    """
    Performing spectral initialization for missing data case.
    """

    T = len(Ys)
    ell = Ys[0].shape[1]
    
    Xs_init = []

    for t in range(T):
        temp = []
        for i in range(ell):
            spect_mat = np.expand_dims( (1 / frac_observed)*Ys[t][:, i], axis=1 )
            temp.append(np.expand_dims( np.linalg.svd( spect_mat )[0][:, 0], axis=1 ))

        Xs_init.append(np.concatenate(temp, axis=1))
        
    return Xs_init


d = 40

# ks = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
ks = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
# Ts = [16, 18, 22, 26, 30, 24, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74]
Ts = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 148, 160, 172]

FRAC_OBSERVED = 0.5


SIGMA = 0.0
NTRIALS = 10
NITER = 10_000

# For AltMin
TOL = 1e-3
N_INNER_ITERS = 5

phase_mat = np.zeros((len(Ts), len(ks)))

for i in range(len(Ts)):
    T = Ts[i]
    
    for j in range(len(ks)):
        k = ks[j]
        print("T:", T, "Rank:", k)
        for trial in range(NTRIALS):

            # Init
            err_U_mm = np.zeros((NITER, ))
            
            # Generate data
            ell = 1
            geo_true, Xs, ts, Gs = generate_data(d=d, k=k, ell=ell, T=T)
            forward, transpose, masks = generate_mc(d=d, ell=ell, T=T, frac_observed=FRAC_OBSERVED)

            # Generate masked data
            Ys = []

            for p, Xi in enumerate(Xs):
                Ys.append( forward(Xi, masks[p]) + np.random.randn(*Xi.shape)*SIGMA)   

            # Initialize data
            Xs_init = spectral_init(Ys=Ys, frac_observed=FRAC_OBSERVED)
            geo_init = init_geodesic(d=d, k=k, Xs=Xs_init, init="rand") # Initialize geodesic

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

                np.savez('phase_matrix_d80_mid.npz', phase_mat)
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
            
            
np.savez('phase_matrix_d80_final.npz', phase_mat)