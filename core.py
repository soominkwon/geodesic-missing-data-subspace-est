#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 22:48:16 2023

@author: soominkwon
"""

import warnings
import numpy as np
import scipy.linalg as la

from utils import check_ortho as _check_ortho
from utils import get_backend as _get_backend


def random_subspace(shape, seed=None):
    # ref: https://math.stackexchange.com/a/3384393
    rng = np.random.default_rng(seed)
    return la.svd(rng.standard_normal(shape), full_matrices=False)[0]


def random_ortho_group(n, seed=None):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n))
    q, r = la.qr(z)
    d = np.diagonal(r, offset=0)
    d.flags.writeable = True  # trying to minimize memory use
    d /= np.abs(d)
    q *= d
    return q


def random_geodesic(shape, length, A0=None, static_k=None, seed=None):
    """Returns the endpoints of a random geodesic on the Grassmannian manifold
    as two orthonormal bases for the subspaces.
    The length of the geodesic may be chosen with the `length` argument.
    The starting endpoint `A0` may also optionally be given
    """
    if (len(shape) != 2) or (shape[0] <= shape[1]):
        raise ValueError(
            f"shape parameter should represent tall matrix, but {shape} was given"
        )
    if A0 is not None and (not _check_ortho(A0) or A0.shape != shape):
        raise ValueError(
            f"The provided A0 is not an orthonormal matrix of shape {shape}"
        )
    rng = np.random.default_rng(seed)
    B = random_subspace(shape, seed=rng)
    A = random_subspace(shape, seed=rng) if A0 is None else A0
    Y, theta, Z = svd_log_map(A, B, np=np, la=la)
    if static_k is not None:
        theta[-static_k:] = 0
    theta /= np.sqrt(np.sum(theta**2))
    if np.any((theta * length) > (np.pi / 2)):
        warnings.warn("length is too large and is wrapping around the Grassmannian")
    C = (A @ (Z * np.cos(theta * length))) + (Y * np.sin(theta * length))
    # avoid any bias from C being a principle direction by random rotation or flip
    C = C @ random_ortho_group(shape[1], seed=rng)
    assert _check_ortho(C)
    return A, C


def random_piecewise_geodesics(shape, lengths, A0=None, seed=None):
    rng = np.random.default_rng(seed)
    segments = [random_geodesic(shape, lengths[0], A0=A0, seed=rng)]
    for length in lengths[1:]:
        segments.append(random_geodesic(shape, length, segments[-1][-1], seed=rng))
    return segments


def sample_subspaces(A, B, N, endpoint=True):
    """Sample subspaces along Grassmannian Geodesic
    at N points between subspaces A and B.

    Given subspaces A and B, a subspace U_t lying a fraction
    t ∈ [0,1] between A and B on a geodesic is given by:
    .. math::
        U_t = AZcos(θt) + Ysin(θt)
            where YSZ' = (I - AA')B(A'B)^{-1}
            and θ = arctan(S)

    Notes
    -----
    While U[0] will represent the same subspace as A
    and U[-1] will represent the same subspace as B (if endpoint=True),
    these are not guaranteed to be the same bases.
    """
    np, la = _get_backend(A)
    d, k = A.shape
    if (d, k) != B.shape:
        raise ValueError(f"A and B do not have the same shape ({A.shape} != {B.shape})")

    Y, theta, Z = svd_log_map(A, B, np=np, la=la)

    U = np.empty((N, d, k))
    for ii, t in enumerate(np.linspace(0.0, 1.0, N, endpoint=endpoint)):
        U[ii] = geodesic_eqn(A, Y, theta, Z, t)

    assert _check_ortho(U), "U is not orthonormal"
    return U


def svd_log_map(A, B, *, np=np, la=la):
    """Given a Grassmanian Geodesic defined by the span
    of orthonormal matrices A & B, returns the necessary
    matrices and principle angles for interpolating to
    a subspace between them on the Grassmannian.
    This is equivalently the svd of log_map(A, B),
    where log_map(A,B) give a matrix in the tangent space of A.
    .. math::
        U_t = AZcos(θt) + Ysin(θt)
    Returns Y, θ, Z
    """
    Y, S, Zh = la.svd((B @ la.inv(A.conj().T @ B)) - A, full_matrices=False)
    # Y, S, Zh = la.svd(la.solve(B.T @ A, B.T).T - A, full_matrices=False)
    Z = Zh.conj().T
    theta = np.arctan(S)
    return Y, theta, Z


def svd_log_map_v2(A, B, *, np=np, la=la):
    """Laura's Formulation
    Note: The principle angles order, and thus corresponding matrix
    columns are reversed relative to the original formulation.
    """
    Z, S, Qh = la.svd(A.conj().T @ B, full_matrices=True)
    Q = Qh.conj().T
    theta = np.arccos(np.clip(S, a_min=0, a_max=1))
    R = B @ Q - A @ (A.conj().T @ (B @ Q))  # = (I-AA')BQ
    U, _, Vh = la.svd(R, full_matrices=False)
    Y = U @ Vh
    return Y, theta, Z


def geodesic_eqn(A, Y, theta, Z=None, t=1.0):
    """aka the inner part of exp_map
    Like exp_map, but tangent direction D is
    given as its components Y,theta,Z instead,
    as these are often already available"""
    # orthonormalizing U shouldn't be required if A, Y, Z are orthonormal
    if Z is not None:
        A = A @ Z
    return (A * np.cos(theta * t)) + (Y * np.sin(theta * t))


def exp_map(A, D, t=1):
    """The Exponential Map on the Grassmannian

    Maps a point D in the tangent space of A to the Grassmannian.

    Parameters
    ----------
    A:
        A starting subspace represented by an dxk orthonormal matrix
    D:
        A dxk tangent matrix at A
    t:
        scale to move along geodesic

    """
    if not _check_ortho(A) or A.shape != D.shape or len(A.shape) != 2:
        raise ValueError(
            "The provided A is not an orthonormal matrix of matching shape"
        )
    if not np.allclose(A.conj().T @ D, 0):
        raise ValueError("Tangent vector D is not tangent to Grassmannian at A")
    Y, theta, Zh = la.svd(D, full_matrices=False)
    Z = Zh.conj().T
    return geodesic_eqn(A, Y, theta, Z, t)


def log_map(A, B):
    """Implements the inverse exponential map on the Grassmannian

    The inverse exponential map (log-map) gives a vector in the
    the tangent space of A that is in the direction of B.
    """
    Y, theta, Z = svd_log_map(A, B)
    return (Y * theta) @ Z.conj().T