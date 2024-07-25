
# -*- coding: utf-8 -*-
__authors__ = 'Francesco Ciucci, Adeleke Maradesa, Baptiste Py'

__date__ = '23th March 2024'

"""
    Required python packages: numpy, scipy.
"""

import numpy as np
from numpy import inf, pi
from numpy.linalg import cholesky
from scipy.optimize import minimize
from math import pi, log, exp,sqrt
#
from . import basics
from . import nearest_PD as nPD
##
# define kernel
def kernel(log_tau, log_tau_prime, sigma_f, ell):

    return (sigma_f**2)*exp(-0.5/(ell**2)*((log_tau-log_tau_prime)**2))

# compute kernel matrix
def compute_K(log_tau_vec, sigma_f, ell):

    N_tau = log_tau_vec.size
    out_K = np.zeros((N_tau, N_tau))

    for m in range(0, N_tau):

        log_tau_m = log_tau_vec[m]

        for n in range(0, N_tau):

            log_tau_n = log_tau_vec[n]
            out_K[m,n] = kernel(log_tau_m, log_tau_n, sigma_f, ell)
    
    out_K = 0.5*(out_K+out_K.T)
    return out_K
##

def compute_A(freq_vec, tau_vec,epsilon,rbf_type, include ="R"):
    
    '''
    """
    compute the discritization matrix A_re and A_im.

    Parameters:
    - freq_vec: vector of frequency
    - tau_vec : vector of timescales
    - epsilon: shape factor 
    - rbf_type: selected RBF type

    Returns:
    - return matrix A
    
    '''
    
    
    N_freqs = freq_vec.size
    N_taus = tau_vec.size
    ### Compute A (putting resistance into consideration)
    if include == "R":
        
        A = np.zeros((2 * N_freqs, N_taus + 1))
        ##
        A_re = basics.assemble_A_re(freq_vec, tau_vec, epsilon, rbf_type)
        A_im = basics.assemble_A_im(freq_vec, tau_vec, epsilon, rbf_type)
        ##
        # Real part
        A[:N_freqs, 0] = 1.0 
        A[:N_freqs, 1:] = A_re
    
        # Imaginary part
        A[N_freqs:, 1:] = A_im
    ### Compute A (putting resistance and inductance into consideration)   
    elif include == "R+L":
        
        A = np.zeros((2*N_freqs, N_taus+2))
        ##
        A_re = basics.assemble_A_re(freq_vec, tau_vec, epsilon, rbf_type)
        A_im = basics.assemble_A_im(freq_vec, tau_vec, epsilon, rbf_type)
#         A_re, A_im = fGP.compute_A_re(freq_vec, tau_vec), fGP.compute_A_im(freq_vec, tau_vec)
        # real part
        A[:N_freqs, 1] = 1.0 
        A[:N_freqs, 2:] = A_re
        # imaginary part
        A[N_freqs:, 0] = 2*pi*freq_vec*1E-4 # normalization by 1E-4
        A[N_freqs:, 2:] = A_im
        
    return A


##GammaL(log_tau_vec,sigma_L,sigma_R,sigma_f, ell)

def compute_Gamma(theta, log_tau_vec, include="R"):
    """
    Compute the Gamma matrix for the case "R+L" or "R".

    Parameters:
    - log_tau_vec: Array of log timescales
    - theta: parameters vector (sigma_L, sigma_R, sigma_f, ell)
    - include: String indicating the inclusion of "R" or "R+L"

    Returns:
    - Gamma: Computed Gamma matrix
    """

    if include == "R":
        #
        sigma_n, sigma_R, sigma_f, ell = theta
        # Compute Gamma for the case "R"
        N_taus = log_tau_vec.size
        K = compute_K(log_tau_vec, sigma_f, ell)
        Gamma = np.zeros((N_taus + 1, N_taus + 1))
        Gamma[0, 0] = sigma_R**2
        Gamma[1:, 1:] = K

    elif include == "R+L":
        #
        sigma_n, sigma_L, sigma_R, sigma_f, ell = theta
        # Compute Gamma for the case "R+L"
        N_taus = log_tau_vec.size
        K = compute_K(log_tau_vec, sigma_f, ell)
        Gamma = np.zeros((N_taus + 2, N_taus + 2))
        Gamma[0, 0] = sigma_L**2
        Gamma[1, 1] = sigma_R**2
        Gamma[2:, 2:] = K

    return Gamma

# calculate the negative marginal log-likelihood (NMLL), including only resistance R.
def NMLL_fct(theta, A, Z_exp_re_im, N_freqs, log_tau_vec):
    
    '''
    Compute the loglikelihood function

    Parameters:
    - theta: parameters vector (sigma_n, sigma_R, sigma_f, ell)
    - Z_exp_re_im: vector of stacked impedance
    - N_freqs: size of frequency vector
    - log_tau_vec: vector of log-timescales

    Returns:
    - NMLL
    '''

    # load the value of the parameters
    sigma_n, sigma_R, sigma_f, ell = theta   

    # Gamma
    Gamma = compute_Gamma(theta, log_tau_vec, include="R")
    # put together the Gamma matrix
    Psi = A@(Gamma@A.T)+(sigma_n**2)*np.eye(2*N_freqs)
    Psi = 0.5*(Psi + Psi.T) # symmetrize
    
    # finding mearest positive definite matrix
    Psi = nPD.nearest_PD(Psi) if not nPD.is_PD(Psi) else Psi

    # Cholesky decomposition
    L = np.linalg.cholesky(Psi)
    
    # solve for alpha
    alpha = np.linalg.solve(L, Z_exp_re_im)
    alpha = np.linalg.solve(L.T, alpha)
    return 0.5*np.dot(Z_exp_re_im,alpha) + np.sum(np.log(np.diag(L)))

## include both resistance and inductance (R+L) in the NMLL
def NMLL_L_fct(theta, A, Z_exp_re_im, N_freqs, log_tau_vec):
    
    '''
    Compute the loglikelihood function

    Parameters:
    - theta: parameters vector (sigma_n, sigma_R, sigma_f, ell)
    - Z_exp_re_im: vector of stacked impedance
    - N_freqs: size of frequency vector
    - log_tau_vec: vector of log-timescales

    Returns:
    - NMLL
    '''

    # load the value of the parameters
    sigma_n, sigma_L, sigma_R, sigma_f, ell = theta
    # Gamma
    Gamma = compute_Gamma(theta, log_tau_vec, include="R+L")
    # put together the Gamma matrix
    Psi = A@(Gamma@A.T)+(sigma_n**2)*np.eye(2*N_freqs)
    Psi = 0.5*(Psi + Psi.T) # symmetrize
    
    # finding nearest positive definite matrix

    Psi = nPD.nearest_PD(Psi) if not nPD.is_PD(Psi) else Psi
        
    #Cholesky decomposition of Psi
    L = np.linalg.cholesky(Psi)
    
    # solve for alpha
    alpha = np.linalg.solve(L, Z_exp_re_im)
    alpha = np.linalg.solve(L.T, alpha)

    return 0.5*np.dot(Z_exp_re_im,alpha) + np.sum(np.log(np.diag(L)))

###
## stacked real and imaginary part of theimpedance 
def Z_exp_stacked(N_freqs,Z_exp):
    Z_exp_re_im = np.zeros(2*N_freqs)
    Z_exp_re_im[:N_freqs] = Z_exp.real
    Z_exp_re_im[N_freqs:] = Z_exp.imag

    return Z_exp_re_im

## extract DRT and recovered impedances
def extract_DRT_and_impedance_vector(N_freqs,samples_gamma,samples_Z_re_im):
    
    '''
    this function extract the recovered DRT and impedances alongside their credible bound

    Parameters:
    - N_freqs: size of frequency vector
    - samples_gamma: DRT values sampled using Hamiltonian Monte Carlo method
    - samples_Z_re_im: the obtained through matrix multiplication i.e., samples_Z_re_im = A@samples  

    Returns:
    - the recovered DRT and impedances alongside their credible bound
    '''
    
    DRT = np.nanmedian(samples_gamma,axis=1)
    lb_bound = np.percentile(samples_gamma, 1, axis=1)
    up_bound = np.percentile(samples_gamma, 99, axis=1)
        ## Re-Im
    Z_re_im_median = np.nanmedian(samples_Z_re_im,axis=1)
        
    ## Recovered Impedance
    Z_re_median = Z_re_im_median[:N_freqs]
    Z_im_median = Z_re_im_median[N_freqs:]
        ### confidence bound
    Z_re_im_percentile_0dot1 = np.percentile(samples_Z_re_im, 1, axis=1)
    Z_re_im_percentile_0dot9 = np.percentile(samples_Z_re_im, 99, axis=1)
        ###
    Z_rec = Z_re_median + 1j*Z_im_median
        
    ### Credible band for the recovered impedance
        
    Zre_lb_bound = Z_re_im_percentile_0dot1[:N_freqs]
    Zre_up_bound = Z_re_im_percentile_0dot9[:N_freqs]
    Zim_lb_bound = Z_re_im_percentile_0dot1[N_freqs:]
    Zim_up_bound = Z_re_im_percentile_0dot9[N_freqs:]
    
    return DRT, lb_bound,up_bound,Z_rec,Zre_lb_bound,Zre_up_bound,Zim_lb_bound,Zim_up_bound