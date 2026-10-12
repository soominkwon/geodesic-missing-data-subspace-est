#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:07:40 2024

@author: soominkwon
"""

import numpy as np
import random


def generate_ms(d, ell, T, m):
    
    def masked_array(data, threshold):
        return (data < threshold).astype(int)

    As = []
    for i in range(T):
        # Generate a mask for each data sample at time t
        As.append( np.random.randn(m, d)*(1/m) )
    
    def forward(x, A):
        return A @ x
        
    def transpose(y, A):
        return A.conj().T @ y

    return forward, transpose, As


def generate_mc(d, ell, T, frac_observed):
    
    def masked_array(data, threshold):
        return (data < threshold).astype(int)

    masks = []
    for i in range(T):
        # Generate a mask for each data sample at time t
        masks.append( np.concatenate( [ masked_array(data=np.random.random((d, 1)), threshold=frac_observed) for l in range(ell) ], axis=1 ) )
    
    def forward(x, mask):
        return mask * x
        
    transpose = forward

    return forward, transpose, masks


def generate_partial_kspace(dims, ell, T, frac_observed):
    """
    Arguments:
        fraction: Fraction of observed of k-space data
    """    
    
    d = dims[0]

    def kspace_mask(d, frac_observed):

        row = d
        mask = np.zeros((d, d))

        total = int(row * frac_observed)

        # For middle k-space sampling
        centric = total // 4
        middle_line = row // 2

        mask[middle_line-centric : middle_line+centric] = 1

        # Sampling the rest
        rest = total - centric
        upper = middle_line - 1
        lower = middle_line + 1

        sample_upper = list( range(upper) )
        random.shuffle( sample_upper )

        del sample_upper[:-(rest // 2)]

        sample_lower = list( range(lower, row-1) )
        random.shuffle( sample_lower )

        del sample_lower[:-(rest // 2)]

        mask[sample_upper] = 1
        mask[sample_lower] = 1

        mask[-1] = 1
    
        return mask
    
    masks = []
    
    for i in range(T):
        masks.append( np.concatenate( [ np.reshape(kspace_mask(d, frac_observed), (-1, 1), order='F') for _ in range(ell) ], axis=1 ) )
        
    def forward(x, masks):  

        forward_x = np.zeros(x.shape, dtype=np.complex64)

        for i in range(ell):
            mask = np.reshape(masks[:, i], dims, order='F')

            forward_x[:, i] = np.reshape( mask * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift( np.reshape(x[:, i], dims, order='F') ), norm="ortho")), (-1, ), order='F')
    
        return forward_x

    def transpose(y, masks=None):
        reverse_y = np.zeros(y.shape, dtype=np.complex64)

        if masks is not None:
            for i in range(ell):
                mask = np.reshape(masks[:, i], dims, order='F')
                reverse_y[:, i] = np.reshape(mask * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift( np.reshape(y[:, i], dims, order='F') ), norm="ortho")), (-1, ), order='F')
        
            return reverse_y
        else:
            for i in range(ell):
                reverse_y[:, i] = np.reshape( np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift( np.reshape(y[:, i], dims, order='F') ), norm="ortho")), (-1, ), order='F')
        
            return reverse_y 


    return forward, transpose, masks


