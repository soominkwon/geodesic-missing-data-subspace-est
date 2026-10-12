#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 18:13:09 2023

@author: soominkwon
"""

import numpy as np
import scipy.linalg as la
from copy import deepcopy

class GeodesicEstimator:
    def __init__(self, rank, ):
        self.rank = rank


    def est_point_tangent(self, geo, data, lambda_Q=0.1):
        """
        Function to estimate HY.
        """
        
        Xs, ts = data["Xs"], data["ts"]
        
        if isinstance(Xs, list):
            HY = np.concatenate([geo.H, geo.Y], axis=1) # current estimate of Q
            
            M_AB = 0.0
            for ti, Xi in zip(ts, Xs):  # parallelizable
                Ui = geo(ti)
                G = Ui.conj().T @ Xi
    
                XG = Xi @ G.conj().T
                XGcos = XG * np.cos(geo.theta * ti)
                XGsin = XG * np.sin(geo.theta * ti)
                M_AB += np.concatenate([XGcos, XGsin], axis=1)
                
            if lambda_Q == 0:
                return 2*M_AB
            else:
                return HY + (2 / lambda_Q) * M_AB
    
    
    def est_theta(self, geo, data, lambda_theta=0.1, iter_inner=5):
        """
        Function to estimate theta.
        """
        
        Xs, ts = data["Xs"], data["ts"]

        r, phi, _ = self.theta_helper(geo.H, geo.Y, Xs)
        t = ts[:, None]
        tt = 2 * t  # two t
        n2tr = -tt * r  # negative two t r, dim(T x k)
        L = tt * n2tr  # Lipschitz constants = -4 t^2 r
    
        # run iterations
        theta = geo.theta
            
        for _ii in range(iter_inner):
            # Single MM step
            arg = tt * theta - phi
            gradf = n2tr * -np.sin(arg)
            denom = np.mod(arg + np.pi, 2 * np.pi) - np.pi
            curvf = tt * gradf
            with np.errstate(divide="ignore", invalid="ignore"):
                curvf /= denom
            m = denom == 0
            curvf[m] = L[m]  # = -4 t^2 r
    
            # theta = theta - gradf.sum(axis=0) / curvf.sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                step = gradf.sum(axis=0) / ( curvf.sum(axis=0) + lambda_theta ) # adding regularizer here
            step[np.isnan(step)] = 0.0  # data is probably zero
    
            theta -= step
            
        return theta
    
    
    def est_data(self, geo, data, lambda_x=0.1, lambda_fn=1.0, L=1, n_inner_loops=1, ):
        
        ts = data["ts"]  
        masks = data["masks"]
        obs = data["measurements"]

        forward, transpose = data["forward"], data["transpose"]
        
        d = data["Xs"][0].shape[0]
                
        for _ in range(n_inner_loops):
    
            for i, ti in enumerate(ts):
                Ui = geo(ti)
                grad_X = transpose(forward(data["Xs"][i], masks[i]) - obs[i], masks[i]) - lambda_fn*(Ui@Ui.conj().T - np.identity(d)) @ data["Xs"][i]
                data["Xs"][i] = data["Xs"][i] - (1 / (L + lambda_x)) * grad_X

        return data


    def fit(self,
            geo,
            data,
            callback, 
            lambda_Q=0.0,
            lambda_theta=0.0,
            lambda_x=0.0,
            lambda_fn=1.0,
            niters=10_000,
            inner_iters=5,
            L=1,
            ):
        
        # Training loop
        for itr in range(niters):
            if callback:
                call_output = callback(itr=itr, geo=geo, data=data)
             
            if call_output:
            	break
        
            # Update Xs        
            data = self.est_data(geo=geo, data=data,
                                          lambda_x=lambda_x, lambda_fn=lambda_fn, L=L)
            #Xs = self.est_data(geo=geo, data=data,
            #                              lambda_x=lambda_x, lambda_fn=lambda_fn, L=L)
            #data["Xs"] = Xs # Update dictionary
            
            # Update H, Y
            for _inner in range(inner_iters):
                M_AB = self.est_point_tangent(geo=geo, data=data, lambda_Q=lambda_Q)
                U, _, Vh = la.svd(M_AB, full_matrices=False)
                C = U @ Vh
                geo.H = C[:, :self.rank]
                geo.Y = C[:, self.rank:]

            geo.theta = self.est_theta(geo=geo, data=data, lambda_theta=lambda_theta, iter_inner=inner_iters)
                
        return geo, data


    def full_fit(self,
            data,
            geo,
            callback, 
            lambda_Q=0.0,
            lambda_theta=0.0,
            niters=10_000,
            inner_iters=5,
            ):
        
        
        # Instantiating parameters    
        Xs, ts = data["Xs"], data["ts"]
        
        # Training loop
        for itr in range(niters):

            # Update H, Y
            for _inner in range(inner_iters):
                M_AB = self.est_point_tangent(geo=geo, Xs=Xs, ts=ts, lambda_Q=lambda_Q)
                U, _, Vh = la.svd(M_AB, full_matrices=False)
                C = U @ Vh
                geo.H = C[:, :self.rank]
                geo.Y = C[:, self.rank:]

            geo.theta = self.est_theta(geo=geo, Xs=Xs, ts=ts, lambda_theta=lambda_theta, iter_inner=inner_iters)
            
            if callback:
                _ = callback(itr=itr, geo=geo, data=Xs)
                
        return geo


    def theta_helper(self, H, Y, Xs):
        """
        Helper function for theta.
        """
        
        T = len(Xs)
    
        if isinstance(Xs, list):
            r = np.empty((T, H.shape[-1]), dtype=H.real.dtype)
            phi = np.empty((T, H.shape[-1]), dtype=H.real.dtype)
            bias = np.empty((T, H.shape[-1]), dtype=H.real.dtype)
            
            # Precompute constants for each t
            for ii, Xi in enumerate(Xs):  # parallelizable
                XiconjT = Xi.conj().T
                XiTatH = XiconjT @ H
                XiTatY = XiconjT @ Y
                a = (XiTatH.conj() * XiTatH).real.sum(axis=0)  # = diag(H.T @ Xi @ Xi.T @ H)
                b = (XiTatY.conj() * XiTatH).real.sum(axis=0)
                c = (XiTatY.conj() * XiTatY).real.sum(axis=0)
                r[ii] = np.sqrt(((a - c) / 2) ** 2 + b**2)
                phi[ii] = np.arctan2(2 * b, (a - c))
                bias[ii] = (a + c) / 2
        else:
            raise Exception("Xs must be in a list.")

        return r, phi, bias
    
