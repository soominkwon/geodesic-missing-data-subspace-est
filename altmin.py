#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:11:17 2024

@author: soominkwon
"""

import numpy as np
from utils import break_data
from copy import deepcopy


# FOR MC
# def solve_L(forward, transpose, Y_concat, L, R, n_inner_iters, tol, mask, lr):    
#     for i in range(n_inner_iters):

#         grad_L = transpose(forward(L@R, mask)- Y_concat, mask) @ R.T
#         L = L - lr*grad_L
        
#         if np.linalg.norm( grad_L ) < tol:
#             break
#     return L


# def solve_R(forward, transpose, Y_concat, L, R, n_inner_iters, tol, mask, lr):    
#     for i in range(n_inner_iters):

#         grad_R = L.T @ transpose(forward(L@R, mask)- Y_concat, mask)
#         R = R - lr*grad_R

#         if np.linalg.norm( grad_R ) < tol:
#             break
#     return R


def solve_L(forward, transpose, Y_concat, L, R, n_inner_iters, tol, mask, lr, ell):    
    
    L = deepcopy(L)
    Ys = break_data(Y_concat, ell)

    for _ in range(n_inner_iters):
        res = []
        X = L@R
        Xs = break_data(X, ell)
        for i in range(len(Xs)):
            res.append( transpose(forward(Xs[i], mask[i])- Ys[i], mask[i]) )
            
        grad_L = np.concatenate(res, axis=1) @ R.T
        L = L - lr*grad_L
        
        if np.linalg.norm( grad_L ) < tol:
            break
    return L


def solve_R(forward, transpose, Y_concat, L, R, n_inner_iters, tol, mask, lr, ell):    
    
    R = deepcopy(R)
    Ys = break_data(Y_concat, ell)

    for _ in range(n_inner_iters):
        res = []
        X = L@R
        Xs = break_data(X, ell)
    
        for i in range(len(Xs)):
            res.append( transpose(forward(Xs[i] , mask[i])- Ys[i], mask[i]) )
            

        grad_R = L.T @ np.concatenate(res, axis=1)
        R = R - lr*grad_R

        if np.linalg.norm( grad_R ) < tol:
            break
    return R


def fit_altmin(Xs_init,
               rank,
               forward,
               transpose,
               mask,
               Y_concat,
               n_inner_iters,
               n_outer_iters,
               tol,
               lr,
               callback=None,
               ell=None):
    
    # Spectral initialization
    L, S, R = np.linalg.svd(np.concatenate(Xs_init, axis=1), full_matrices=False)
    L_init = L[:, :rank] @ np.diag(S[:rank])**(1/2)
    R_init = np.diag(S[:rank])**(1/2) @ R[:rank]
    
    
    # Training loop
    for itr in range(n_outer_iters):
        # L update
        L_init = solve_L(forward=forward, transpose=transpose, Y_concat=Y_concat, L=L_init, R=R_init, 
                         n_inner_iters=n_inner_iters, tol=tol, mask=mask, lr=lr, ell=ell)
        # R update
        R_init = solve_R(forward=forward,  transpose=transpose, Y_concat=Y_concat, L=L_init, R=R_init, 
                         n_inner_iters=n_inner_iters, tol=tol, mask=mask, lr=lr, ell=ell)

        if callback:
            feedback = callback(itr=itr, data=L_init@R_init)
            
        if feedback:
            break
            
    return L_init, R_init