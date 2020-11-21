# coding:utf-8
"""
@auther tzw
@ref:
https://github.com/maggie0106/Graph-CNN-in-3D-Point-Cloud-Classification/blob/master/global_pooling_model/utils.py
"""
import os, sys, time
import numpy as np
import scipy
import scipy.sparse as ss
from scipy.sparse.linalg import eigsh

def adjacency(dist, idx):
    """Return the adjacency matrix of a kNN graph."""
    M, k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0
    # Weights.
    sigma2 = np.mean(dist[:, -1]) ** 2
    #print sigma2
    dist = np.exp(- dist ** 2 / sigma2)
    print("sigma2",sigma2,sigma2.shape)

    # Weight matrix.
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M * k)
    V = dist.reshape(M * k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))
    # No self-connections.
    W.setdiag(0)

    # Non-directed graph.
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)
    return W

##################################################
# *****  *****
##################################################
def create_laplacian(W, normalize=True):
    """
    @W: adjacency matrix
    @normalize: if False, this function return combinatorial lapalcian
    @return: laplacian
    https://github.com/zEttOn86/Convolutional-Neural-Networks-on-Graphs-with-Fast-Localized-Spectral-Filtering/blob/master/lib/graph.py
    """
    n = W.shape[0]
    W = ss.csr_matrix(W)
    WW_diag = W.dot(ss.csr_matrix(np.ones((n, 1)))).todense()
    if normalize:
        WWds = np.sqrt(WW_diag)
        # Let the inverse of zero entries become zero.
        WWds[WWds == 0] = np.float("inf")
        WW_diag_invroot = 1. / WWds
        D_invroot = ss.lil_matrix((n, n))
        D_invroot.setdiag(WW_diag_invroot)
        D_invroot = ss.csr_matrix(D_invroot)
        I = scipy.sparse.identity(W.shape[0], format='csr', dtype=W.dtype)
        L = I - D_invroot.dot(W.dot(D_invroot))
    else:
        D = ss.lil_matrix((n, n))
        D.setdiag(WW_diag)
        D = ss.csr_matrix(D)
        L = D - W

    return L.astype(W.dtype)
    #return L

def create_scaled_laplacian(adj):
    """
    @adj: adjacency matrix
    @return: scaled laplacian (dtype = float32)
    """
    laplacian = create_laplacian(adj)
    # Calc eigen value
    largest_eigval, _ = scipy.sparse.linalg.eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - scipy.sparse.identity(laplacian.shape[0], format='csr', dtype=laplacian.dtype)
    return scaled_laplacian

##################################################
# ***** I dont use below functions *****
##################################################

def normalize_adj(adj):
    """
    * https://github.com/zEttOn86/Graph-CNN-in-3D-Point-Cloud-Classification/blob/master/utils/mathematical_functions.py
    """
    adj = scipy.sparse.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def scaled_laplacian(adj): # future work
    """
    * https://github.com/zEttOn86/Graph-CNN-in-3D-Point-Cloud-Classification/blob/master/utils/mathematical_functions.py
    """
    adj_normalized = normalize_adj(adj)
    # Create normalized laplacian
    laplacian = scipy.sparse.eye(adj.shape[0]) - adj_normalized
    # Calc largest eigen value
    largest_eigval, _ = scipy.sparse.linalg.eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - scipy.sparse.eye(adj.shape[0])
    return scaled_laplacian
