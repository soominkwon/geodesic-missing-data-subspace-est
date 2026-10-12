#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 23:04:47 2026

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
        Vectorized estimation of HY.
        Speeds up computation by stacking Xs into a (T, N, D) tensor.
        """
        Xs, ts = data["Xs"], data["ts"]
        
        # 1. Stack data for vectorization: List[(N, D)] -> Array(T, N, D)
        # Note: This assumes all Xi in Xs have the same shape.
        Xs_stack = np.stack(Xs) 
        
        # 2. Vectorize Geodesic Point Computation: U(t)
        # Assuming standard geodesic: U(t) = H cos(Θt) + Y sin(Θt)
        # Shapes: H, Y are (N, K); theta is (K,); t is (T,)
        
        # Prepare angles: (T, 1, K) for broadcasting
        angles = (ts[:, None] * geo.theta[None, :]) 
        cos_t = np.cos(angles)
        sin_t = np.sin(angles)
        
        # Compute all Ui at once: (T, N, K)
        # geo.H[None, :, :] broadcasts H to (1, N, K)
        Ui_stack = (geo.H[None, :, :] * cos_t[:, None, :]) + \
                   (geo.Y[None, :, :] * sin_t[:, None, :])
        
        # 3. Vectorize G calculation
        # G = Ui.conj().T @ Xi
        # (T, K, N) @ (T, N, D) -> (T, K, D)
        G_stack = np.matmul(Ui_stack.transpose(0, 2, 1).conj(), Xs_stack)
        
        # 4. Vectorize XG calculation
        # XG = Xi @ G.conj().T
        # (T, N, D) @ (T, D, K) -> (T, N, K)
        XG_stack = np.matmul(Xs_stack, G_stack.transpose(0, 2, 1).conj())
        
        # 5. Apply trig scaling and sum
        # Broadcast cos/sin (T, 1, K) against XG (T, N, K)
        XGcos = XG_stack * cos_t[:, None, :]
        XGsin = XG_stack * sin_t[:, None, :]
        
        # Sum over time axis (axis 0)
        sum_XGcos = np.sum(XGcos, axis=0)
        sum_XGsin = np.sum(XGsin, axis=0)
        
        M_AB = np.concatenate([sum_XGcos, sum_XGsin], axis=1)

        HY = np.concatenate([geo.H, geo.Y], axis=1)

        if lambda_Q == 0:
            return 2 * M_AB
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
            Vectorized helper for theta.
            Eliminates the loop over T data points.
            """
            # Stack Xs: (T, N, D)
            if isinstance(Xs, list):
                Xs_stack = np.stack(Xs)
            else:
                Xs_stack = Xs
    
            # Precompute transposed conjugates: (T, D, N)
            Xs_H = Xs_stack.transpose(0, 2, 1).conj()
    
            # Batch Matmul: (T, D, N) @ (N, K) -> (T, D, K)
            # We treat H and Y as (1, N, K) for broadcasting or just rely on matmul behavior
            XiTatH = np.matmul(Xs_H, H) 
            XiTatY = np.matmul(Xs_H, Y)
    
            # Compute diagonal sums (real parts of dot products)
            # equivalent to: (A.conj() * B).real.sum(axis=0) where A, B are vectors
            # Here we want the sum over the dimension D (axis 1)
            
            # (T, D, K) * (T, D, K) -> sum over D -> (T, K)
            a = (XiTatH.conj() * XiTatH).real.sum(axis=1)
            b = (XiTatY.conj() * XiTatH).real.sum(axis=1)
            c = (XiTatY.conj() * XiTatY).real.sum(axis=1)
    
            # The rest is element-wise and fast
            r = np.sqrt(((a - c) / 2) ** 2 + b**2)
            phi = np.arctan2(2 * b, (a - c))
            bias = (a + c) / 2
    
            return r, phi, bias
        
