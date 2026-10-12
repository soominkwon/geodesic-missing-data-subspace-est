#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 22:49:09 2023

@author: soominkwon
"""

import numpy as np
import scipy.linalg as la
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import torch


try:
    import cupy
    import cupy.linalg as cupylinalg
except ImportError:
    cupy = None
    cupylinalg = None


def get_backend(x):
    if "cupy" in type(x).__module__:
        backend = cupy, cupylinalg
    else:
        backend = np, la
    return backend


def check_ortho(A, B=None):
    """Checks that the columns of the given matrix are mutually orthonormal
    via A.conj().T@A = I
    """
    np, la = get_backend(A)
    d, k = A.shape[-2:]
    if B is None:
        return np.allclose(np.round(np.moveaxis(A, -1, -2).conj() @ A, 3), np.eye(k))
    else:
        return np.allclose(np.round(np.moveaxis(A, -1, -2).conj() @ B, 2), 0.0)


def projection_mse(A, B):
    """The squared projection distance divided by k.
    A subspace error that equals the sum of squared sines
    of principle angles:
    .. math::
        U, S, V' = svd(A'B)
        θ = arccos(S)
        err(A,B)  = (1/k)sum(sin(θi)**2)
                  = (1/k)norm((I - AA')B)**2
                  = (1/k)(k - norm(A'B)**2)
                  = (1/2k)norm(AA' - BB')**2

    This error satisfies symmetry,
    and is only zero if A=B.
    I do not know if it satisfies the triangle inequality.
    """

    return np.sum((B - A @ (np.moveaxis(A, -1, -2).conj() @ B)) ** 2) / A.shape[-1]


def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    
    #sub_img = (img - np.min(img))
    #return sub_img / np.max(sub_img)
    min_val = np.min(img)
    max_val = np.max(img)
    
    return (img - min_val) / (max_val - min_val)

def compute_PSNR(Xs_est, Xs_true, dim=None):
    if isinstance(Xs_est, list):
        Xs_est = np.concatenate(Xs_est, axis=1)
    if isinstance(Xs_true, list):
        Xs_true = np.concatenate(Xs_true, axis=1)
    
    psnr = 0
    
    T = Xs_est.shape[1]

    for i in range(T):
        psnr += peak_signal_noise_ratio( normalize_np(Xs_true[:, i]), 
                                         normalize_np(Xs_est[:, i]) )
        
    return psnr / T
        
    
def compute_SSIM(Xs_est, Xs_true, dim=None):
    if isinstance(Xs_est, list):
        Xs_est = np.concatenate(Xs_est, axis=1)
    if isinstance(Xs_true, list):
        Xs_true = np.concatenate(Xs_true, axis=1)
    
    ssim_score = 0
    
    T = Xs_est.shape[1]

    for i in range(T):
        ssim_score += ssim( normalize_np(Xs_true[:, i]), 
                            normalize_np(Xs_est[:, i]),
                            data_range=1.0)
        
    return ssim_score / T

        
def compute_RMSE(Xs_est, Xs_true, dim=None):
    if isinstance(Xs_est, list):
        Xs_est = np.concatenate(Xs_est, axis=1)
    if isinstance(Xs_true, list):
        Xs_true = np.concatenate(Xs_true, axis=1)
    
    mse_score = 0
    
    T = Xs_est.shape[1]
    for i in range(T):
        mse_score += mse( normalize_np(Xs_true[:, i]), 
                            normalize_np(Xs_est[:, i]) )
        
    return np.sqrt( mse_score / T )


def break_data(data, ell):
    """
    Break data of (d, frames) into a list of (d, ell) matrices.
    """

    frames = data.shape[1]

    data_list = []

    st = 0
    en = ell
    for i in range(frames // ell):
        data_list.append(data[:, st:en])
        
        st += ell
        en += ell

    return data_list