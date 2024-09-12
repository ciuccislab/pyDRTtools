# -*- coding: utf-8 -*-
__authors__ = 'Francesco Ciucci, Ting Hei Wan, Baptiste Py, Adeleke Maradesa'

__date__ = '10th April 2024'


import numpy as np
from numpy import exp
from math import pi, log, sqrt
from scipy import integrate
from scipy.optimize import fsolve
from scipy.linalg import toeplitz
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

from numpy import *
from scipy.optimize import minimize
from . import parameter_selection as param

"""
This file stores all the functions that are shared by all three DRT methods, i.e., simple, Bayesian, and Bayesian Hilbert Transform.
References: 
    [1] T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, Influence of the discretization methods on the distribution of relaxation times deconvolution: Implementing radial basis functions with DRTtools, Electrochimica Acta. 184 (2015) 483-499.
    [2] M. Saccoccio, T. H. Wan, C. Chen,F. Ciucci, Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: Ridge and lasso regression methods - A theoretical and experimental study, Electrochimica Acta. 147 (2014) 470-482.
    [3] J. Liu, T. H. Wan, F. Ciucci, A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores, Electrochimica Acta. 357 (2020) 136864.
    [4] A. Maradesa, B. Py, T.H. Wan, M.B. Effat, F. Ciucci, Selecting the regularization parameter in the distribution of relaxation times, Journal of the Electrochemical Society. 170 (2023) 030502.
"""


# Part 1: Discretization of the DRT problem and recovery of the DRT using ridge regression
def g_i(freq_n, tau_m, epsilon, rbf_type):
    
    """ 
       This function generates the elements of A_re based on the radial-basis-function (RBF) expansion
       Inputs: 
            freq_n: frequency
            tau_m: log timescale (log(1/freq_m))
            epsilon : shape factor of radial basis functions used for discretization
            rbf_type: selected RBF type
       Outputs:
            Elements of A_re based on the RBF expansion
    """
    
    alpha = 2*pi*freq_n*tau_m  
    
    rbf_switch = {
                'Gaussian': lambda x: exp(-(epsilon*x)**2),
                'C0 Matern': lambda x: exp(-abs(epsilon*x)),
                'C2 Matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                'C4 Matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                'C6 Matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                'Inverse Quadratic': lambda x: 1/(1+(epsilon*x)**2),
                'Inverse Quadric': lambda x: 1/sqrt(1+(epsilon*x)**2),
                'Cauchy': lambda x: 1/(1+abs(epsilon*x))
                }
    
    rbf = rbf_switch.get(rbf_type)
    integrand_g_i = lambda x: 1./(1.+(alpha**2)*exp(2.*x))*rbf(x) # see equation (32) in [1]
    out_val = integrate.quad(integrand_g_i, -50, 50, epsabs=1E-9, epsrel=1E-9)
    
    return out_val[0]


def g_ii(freq_n, tau_m, epsilon, rbf_type):
    
    """
       This function generates the elements of A_im based on the RBF expansion
       Inputs:
           freq_n :frequency
           tau_m : log timescale (log(1/freq_m))
           epsilon  : shape factor of radial basis functions used for discretization
           rbf_type : selected RBF type    
       Outputs:
           Elements of A_im based on the RBF expansion
    """ 
    
    alpha = 2*pi*freq_n*tau_m  
    
    rbf_switch = {
                'Gaussian': lambda x: exp(-(epsilon*x)**2),
                'C0 Matern': lambda x: exp(-abs(epsilon*x)),
                'C2 Matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                'C4 Matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                'C6 Matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                'Inverse Quadratic': lambda x: 1/(1+(epsilon*x)**2),
                'Inverse Quadric': lambda x: 1/sqrt(1+(epsilon*x)**2),
                'Cauchy': lambda x: 1/(1+abs(epsilon*x))
                }
    
    rbf = rbf_switch.get(rbf_type)
    integrand_g_ii = lambda x: alpha/(1./exp(x)+(alpha**2)*exp(x))*rbf(x) # see (33) in [1]
    out_val = integrate.quad(integrand_g_ii, -50, 50, epsabs=1E-9, epsrel=1E-9)
    
    return out_val[0]

def compute_epsilon(freq, coeff, rbf_type, shape_control): 
    
    """
       This function is used to compute the shape factor of the radial basis functions used for discretization. 
       Inputs:
            freq: frequency
            coeff: scalar such that the full width at half maximum (FWHM) of the RBF is equal to 1/coeff times the average relaxation time spacing in logarithm scale
            rbf_type: selected RBF type 
            shape_control: shape of the RBF, which is set with either the coefficient, or with the option "shape factor" through the shape factor 𝜇
       Output: 
           epsilon (shape factor of radial basis functions used for discretization)
    """ 
    
    N_freq = freq.shape[0]
    
    if rbf_type == 'PWL':
        return 0
    
    rbf_switch = {
                'Gaussian': lambda x: exp(-(x)**2)-0.5,
                'C0 Matern': lambda x: exp(-abs(x))-0.5,
                'C2 Matern': lambda x: exp(-abs(x))*(1+abs(x))-0.5,
                'C4 Matern': lambda x: 1/3*exp(-abs(x))*(3+3*abs(x)+abs(x)**2)-0.5,
                'C6 Matern': lambda x: 1/15*exp(-abs(x))*(15+15*abs(x)+6*abs(x)**2+abs(x)**3)-0.5,
                'Inverse Quadratic': lambda x: 1/(1+(x)**2)-0.5,
                'Inverse Quadric': lambda x: 1/sqrt(1+(x)**2)-0.5,
                'Cauchy': lambda x: 1/(1+abs(x))-0.5
                }

    rbf = rbf_switch.get(rbf_type)
    
    if shape_control == 'FWHM Coefficient': # equivalent as the 'FWHM Coefficient' option in the Matlab code
    
        FWHM_coeff = 2*fsolve(rbf,1)[0]
        delta = np.mean(np.diff(np.log(1/freq.reshape(N_freq)))) # see (13) in [1]
        epsilon = coeff*FWHM_coeff/delta
        
    else: # equivalent as the 'Shape Factor' option in the Matlab code

        epsilon = coeff
    
    return epsilon
    

# Approximation matrix of the DRT for the real and imaginary parts of the EIS data

def inner_prod_rbf_1(freq_n, freq_m, epsilon, rbf_type):
    
    """ 
       This function computes the inner product of the first derivatives of the RBFs 
       with respect to tau_n=log(1/freq_n) and tau_m = log(1/freq_m)
       Inputs: 
           freq_n: frequency
           freq_m: frequency 
           epsilon: shape factor 
           rbf_type: selected RBF type
       Outputs: 
           norm of the first derivative of the RBFs with respect to log(1/freq_n) and log(1/freq_m)
    """  
    
    a = epsilon*log(freq_n/freq_m)
    
    if rbf_type == 'Inverse Quadric':
        y_n = -log(freq_n)
        y_m = -log(freq_m)
        
        # could only find numerical version
        rbf_n = lambda y: 1/sqrt(1+(epsilon*(y-y_n))**2)
        rbf_m = lambda y: 1/sqrt(1+(epsilon*(y-y_m))**2)
        
        # compute derivative
        delta = 1E-8
        sqr_drbf_dy = lambda y: 1/(2*delta)*(rbf_n(y+delta)-rbf_n(y-delta))*1/(2*delta)*(rbf_m(y+delta)-rbf_m(y-delta))
        
        # out_IP = integral(@(y) sqr_drbf_dy(y),-Inf,Inf);    
        out_val = integrate.quad(sqr_drbf_dy, -50, 50, epsabs=1E-9, epsrel=1E-9)
        out_val = out_val[0]
        
    elif rbf_type == 'Cauchy':
        if a == 0:
            out_val = 2/3*epsilon
        else:
            num = abs(a)*(2+abs(a))*(4+3*abs(a)*(2+abs(a)))-2*(1+abs(a))**2*(4+abs(a)*(2+abs(a)))*log(1+abs(a))
            den = abs(a)**3*(1+abs(a))*(2+abs(a))**3
            out_val = 4*epsilon*num/den
        
    else:
        rbf_switch = {
                    'Gaussian': -epsilon*(-1+a**2)*exp(-(a**2/2))*sqrt(pi/2),
                    'C0 Matern': epsilon*(1-abs(a))*exp(-abs(a)),
                    'C2 Matern': epsilon/6*(3+3*abs(a)-abs(a)**3)*exp(-abs(a)),
                    'C4 Matern': epsilon/30*(105+105*abs(a)+30*abs(a)**2-5*abs(a)**3-5*abs(a)**4-abs(a)**5)*exp(-abs(a)),
                    'C6 Matern': epsilon/140*(10395 +10395*abs(a)+3780*abs(a)**2+315*abs(a)**3-210*abs(a)**4-84*abs(a)**5-14*abs(a)**6-abs(a)**7)*exp(-abs(a)),
                    'Inverse Quadratic': 4*epsilon*(4-3*a**2)*pi/((4+a**2)**3)
                    }
        out_val = rbf_switch.get(rbf_type)
        
    return out_val


def inner_prod_rbf_2(freq_n, freq_m, epsilon, rbf_type):
    
    """ 
       This function computes the inner product of the second derivatives of the RBFs 
       with respect to tau_n=log(1/freq_n) and tau_m = log(1/freq_m)
       Inputs: 
           freq_n: frequency
           freq_m: frequency 
           epsilon: shape factor 
           rbf_type: selected RBF type
       Outputs: 
           norm of the second derivative of the RBFs with respect to log(1/freq_n) and log(1/freq_m)
    """    
    
    a = epsilon*log(freq_n/freq_m)
    
    if rbf_type == 'Inverse Quadric':
        y_n = -log(freq_n)
        y_m = -log(freq_m)
        
        # could only find numerical version
        rbf_n = lambda y: 1/sqrt(1+(epsilon*(y-y_n))**2)
        rbf_m = lambda y: 1/sqrt(1+(epsilon*(y-y_m))**2)
        
        # compute derivative
        delta = 1E-4
        sqr_drbf_dy = lambda y: 1/(delta^2)*(rbf_n(y+delta)-2*rbf_n(y)+rbf_n(y-delta))*1/(delta^2)*(rbf_m(y+delta)-2*rbf_m(y)+rbf_m(y-delta))
        
        # Matlab code: out_IP = integral(@(y) sqr_drbf_dy(y),-Inf,Inf);    
        out_val = integrate.quad(sqr_drbf_dy, -50, 50, epsabs=1E-9, epsrel=1E-9)
        out_val = out_val[0]
        
    elif rbf_type == 'Cauchy':
        if a == 0:
            out_val = 8/5*epsilon**3
        else:
            num = abs(a)*(2+abs(a))*(-96 +abs(a)*(2+abs(a))*(-30 +abs(a)*(2+abs(a)))*(4+abs(a)*(2+abs(a))))+\
                  12*(1+abs(a))^2*(16+abs(a)*(2+abs(a))*(12+abs(a)*(2+abs(a))))*log(1+abs(a))
            den = abs(a)^5*(1+abs(a))*(2+abs(a))**5
            out_val = 8*epsilon^3*num/den
        
    else:
        rbf_switch = {
                    'Gaussian': epsilon**3*(3-6*a**2+a**4)*exp(-(a**2/2))*sqrt(pi/2),
                    'C0 Matern': epsilon**3*(1+abs(a))*exp(-abs(a)),
                    'C2 Matern': epsilon**3/6*(3 +3*abs(a)-6*abs(a)**2+abs(a)**3)*exp(-abs(a)),
                    'C4 Matern': epsilon**3/30*(45 +45*abs(a)-15*abs(a)**3-5*abs(a)**4+abs(a)**5)*exp(-abs(a)),
                    'C6 Matern': epsilon**3/140*(2835 +2835*abs(a)+630*abs(a)**2-315*abs(a)**3-210*abs(a)**4-42*abs(a)**5+abs(a)**7)*exp(-abs(a)),
                    'Inverse Quadratic': 48*(16 +5*a**2*(-8 + a**2))*pi*epsilon**3/((4 + a**2)**5)
                    }
        out_val = rbf_switch.get(rbf_type)
        
    return out_val


def gamma_to_x(gamma_vec, tau_vec, epsilon, rbf_type):
    
    """  
       This function maps the gamma_vec back to the x vector (x = gamma for piecewise linear functions) 
       Inputs:
            gamma_vec : DRT vector
            tau_vec : vector of log timescales (tau = log(1/frequency))
            epsilon: shape factor 
            rbf_type: selected RBF type
       Outputs:
            x_vec obtained by mapping gamma_vec to x = gamma
    """  
    
    if rbf_type == 'PWL':
        x_vec = gamma_vec
        
    else:
        rbf_switch = {
                'Gaussian': lambda x: exp(-(epsilon*x)**2),
                'C0 Matern': lambda x: exp(-abs(epsilon*x)),
                'C2 Matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                'C4 Matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                'C6 Matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                'Inverse Quadratic': lambda x: 1/(1+(epsilon*x)**2),
                'Inverse Quadric': lambda x: 1/sqrt(1+(epsilon*x)**2),
                'Cauchy': lambda x: 1/(1+abs(epsilon*x))
                }
        
        rbf = rbf_switch.get(rbf_type)
        
        N_taus = tau_vec.size
        B = np.zeros([N_taus, N_taus])
        
        for p in range(0, N_taus):
            for q in range(0, N_taus):
                delta_log_tau = log(tau_vec[p])-log(tau_vec[q])
                B[p,q] = rbf(delta_log_tau)
                
        B = 0.5*(B+B.T)
                
        x_vec = np.linalg.solve(B, gamma_vec)
            
    return x_vec


def x_to_gamma(x_vec, tau_map_vec, tau_vec, epsilon, rbf_type): 
    
    """  
       This function maps the x vector to the gamma_vec
       Inputs:
            x_vec : the DRT vector obtained by mapping gamma_vec to x
            tau_map_vec : log(1/frequency) vector mapping x_vec to gamma_vec
            tau_vec : log(1/frequency) vector
            epsilon: shape factor 
            rbf_type: selected RBF type
       Outputs: 
            tau_vec and gamma_vec obtained by mapping x to gamma
    """
    
    if rbf_type == 'PWL':
        gamma_vec = x_vec
        out_tau_vec = tau_vec    

    else:
        rbf_switch = {
                    'Gaussian': lambda x: exp(-(epsilon*x)**2),
                    'C0 Matern': lambda x: exp(-abs(epsilon*x)),
                    'C2 Matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                    'C4 Matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                    'C6 Matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                    'Inverse Quadratic': lambda x: 1/(1+(epsilon*x)**2),
                    'Inverse Quadric': lambda x: 1/sqrt(1+(epsilon*x)**2),
                    'Cauchy': lambda x: 1/(1+abs(epsilon*x))
                    }
        
        rbf = rbf_switch.get(rbf_type)
        
        N_taus = tau_vec.size
        N_tau_map = tau_map_vec.size
        gamma_vec = np.zeros([N_tau_map, 1])

        B = np.zeros([N_tau_map, N_taus])
        
        for p in range(0, N_tau_map):
            for q in range(0, N_taus):
                delta_log_tau = log(tau_map_vec[p])-log(tau_vec[q])
                B[p,q] = rbf(delta_log_tau)              
                
        gamma_vec = B@x_vec
        out_tau_vec = tau_map_vec 
        
    return out_tau_vec, gamma_vec

# compute A_re matrix
def assemble_A_re(freq_vec, tau_vec, epsilon, rbf_type):
    """
       This function computes the discretization matrix, A_re, used to compute the real part of the impedance
       Inputs:
            freq_vec: vector of frequencies
            tau_vec: vector of timescales
            epsilon: shape factor 
            rbf_type: selected RBF type
            flag1: nature of the run, i.e.i, Simple or BHT run
            flag2: nature of the data, i.e., impedance or admittance, for the BHT run
       Output: 
            Approximation matrix A_re
    """    

    # compute omega and the number of frequencies and timescales tau
    omega_vec = 2. * pi * freq_vec
    N_freqs, N_taus = freq_vec.size, tau_vec.size

    # define the A_re output matrix
    out_A_re = np.zeros((N_freqs, N_taus))
    
    # # Calculate the standard deviation of logarithmic differences between consecutive timescales (tau = 1/freq_vec)
    std_diff_freq = np.std(np.diff(np.log(1 / freq_vec)))
    # Calculate the mean of logarithmic differences between consecutive timescales (tau = 1/freq_vec)
    mean_diff_freq = np.mean(np.diff(np.log(1 / freq_vec)))

    # check if the frequencies are sufficiently log spaced and that N_freqs = N_taus
    toeplitz_trick = std_diff_freq / mean_diff_freq < 0.01 and N_freqs == N_taus

    if toeplitz_trick and rbf_type != 'PWL':     ## use toeplitz trick
        R = np.array([g_i(freq_vec[0], tau, epsilon, rbf_type) for tau in tau_vec])
        C = np.array([g_i(freq, tau_vec[0], epsilon, rbf_type) for freq in freq_vec])
        out_A_re = toeplitz(C, R)

    else:
        for p in range(N_freqs):
            for q in range(N_taus):
                if rbf_type == 'PWL':  # see (A.3a) and (A.4) in [2]  
                    if q == 0:
                        out_A_re[p, q] = 0.5 / (1 + (omega_vec[p] * tau_vec[q]) ** 2) * log(tau_vec[q + 1] / tau_vec[q])
                    elif q == N_taus - 1:
                        out_A_re[p, q] = 0.5 / (1 + (omega_vec[p] * tau_vec[q]) ** 2) * log(tau_vec[q] / tau_vec[q - 1])
                    else:
                        out_A_re[p, q] = 0.5 / (1 + (omega_vec[p] * tau_vec[q]) ** 2) * log(tau_vec[q + 1] / tau_vec[q - 1])
                else:
                    out_A_re[p, q] = g_i(freq_vec[p], tau_vec[q], epsilon, rbf_type)

    return out_A_re


# compute A_im matrix
def assemble_A_im(freq_vec, tau_vec, epsilon, rbf_type):
    
    """
       This function computes the discretization matrix, A_im, for the imaginary part of the impedance
       Inputs:
            freq_vec: vector of frequencies
            tau_vec: vector of timescales
            epsilon: shape factor 
            rbf_type: selected RBF type
            flag1: nature of the run, i.e.i, simple or BHT run
            flag2: nature of the data, i.e., impedance or admittance, for the BHT run
       Output: 
            Approximation matrix A_im
    """ 

    # compute omega and the number of frequencies and timescales tau
    omega_vec = 2. * pi * freq_vec
    N_freqs, N_taus = freq_vec.size, tau_vec.size

    # define the A_re output matrix
    out_A_im = np.zeros((N_freqs, N_taus))

    # Calculate the standard deviation of logarithmic differences between consecutive timescales (tau = 1/freq_vec)
    std_diff_freq = np.std(np.diff(np.log(1 / freq_vec)))
    # Calculate the mean of logarithmic differences between consecutive timescales (tau = 1/freq_vec)
    mean_diff_freq = np.mean(np.diff(np.log(1 / freq_vec)))

    # check if the frequencies are sufficiently log spaced and that N_freqs = N_taus
    toeplitz_trick = std_diff_freq / mean_diff_freq < 0.01 and N_freqs == N_taus


    if toeplitz_trick and rbf_type != 'PWL':
        R = np.array([-g_ii(freq_vec[0], tau, epsilon, rbf_type) for tau in tau_vec])
        C = np.array([-g_ii(freq, tau_vec[0], epsilon, rbf_type) for freq in freq_vec])
        out_A_im = toeplitz(C, R)

    else:                         # see (A.3b) and (A.5) in [2]
        for p in range(N_freqs):
            for q in range(N_taus):
                if rbf_type == 'PWL':
                    if q == 0:
                        out_A_im[p, q] = -0.5 * (omega_vec[p] * tau_vec[q]) / (
                                    1 + (omega_vec[p] * tau_vec[q]) ** 2) * log(tau_vec[q + 1] / tau_vec[q])
                    elif q == N_taus - 1:
                        out_A_im[p, q] = -0.5 * (omega_vec[p] * tau_vec[q]) / (
                                    1 + (omega_vec[p] * tau_vec[q]) ** 2) * log(tau_vec[q] / tau_vec[q - 1])
                    else:
                        out_A_im[p, q] = -0.5 * (omega_vec[p] * tau_vec[q]) / (
                                    1 + (omega_vec[p] * tau_vec[q]) ** 2) * log(tau_vec[q + 1] / tau_vec[q - 1])
                else:
                    out_A_im[p, q] = -g_ii(freq_vec[p], tau_vec[q], epsilon, rbf_type)

    return out_A_im


def assemble_M_1(tau_vec, epsilon, rbf_type):   # see (38) in [1]

    """
       This function computes the matrix, M, of the inner products of the first derivatives of the RBF functions used in 
       the expansion. 
       Inputs:
            tau_vec: vector of timescales
            epsilon: shape factor 
            rbf_type: selected RBF type
            flag: nature of the run, i.e.i, simple or BHT run
       Output: 
            Matrix M
    """

    freq_vec = 1 / tau_vec
    N_taus = tau_vec.size

    # define the M output matrix
    out_M = np.zeros([N_taus, N_taus])

    # Compute the standard deviation and mean of the logarithmic differences between consecutive elements in tau_vec
    std_diff_freq = np.std(np.diff(np.log(tau_vec)))
    mean_diff_freq = np.mean(np.diff(np.log(tau_vec)))

    # if they are, we apply the toeplitz trick
    toeplitz_trick = std_diff_freq / mean_diff_freq < 0.01

    if toeplitz_trick and rbf_type != 'PWL':  # apply the toeplitz trick to compute the M matrix

        R = np.array([inner_prod_rbf_1(freq_vec[0], freq_vec[n], epsilon, rbf_type) for n in range(N_taus)])
        C = np.array([inner_prod_rbf_1(freq_vec[n], freq_vec[0], epsilon, rbf_type) for n in range(N_taus)])

        out_M = toeplitz(C, R)

    elif rbf_type == 'PWL':

        out_L_temp = np.zeros([N_taus - 1, N_taus])

        for iter_freq_n in range(N_taus - 1):
            delta_loc = log((1 / freq_vec[iter_freq_n + 1]) / (1 / freq_vec[iter_freq_n]))
            out_L_temp[iter_freq_n, iter_freq_n] = -1 / delta_loc
            out_L_temp[iter_freq_n, iter_freq_n + 1] = 1 / delta_loc

        out_M = out_L_temp.T @ out_L_temp

    else:  # compute rbf with brute force

        for n in range(N_taus):
            for m in range(N_taus):
                out_M[n, m] = inner_prod_rbf_1(freq_vec[n], freq_vec[m], epsilon, rbf_type)

    return out_M


def assemble_M_2(tau_vec, epsilon, rbf_type):   # see (38) in [1]

    """
       This function computes the matrix, M, of the inner products of the second derivatives of the RBF functions used in 
       the expansion. 
       Inputs:
            tau_vec: vector of timescales
            epsilon: shape factor 
            rbf_type: selected RBF type
            flag: nature of the run, i.e.i, simple or BHT run
       Output: 
            Matrix M
    """ 

    freq_vec = 1 / tau_vec
    N_taus = tau_vec.size

    # define the M output matrix
    out_M = np.zeros([N_taus, N_taus])

    # Compute the standard deviation and mean of the logarithmic differences between consecutive elements in tau_vec
    std_diff_freq = np.std(np.diff(np.log(tau_vec)))
    mean_diff_freq = np.mean(np.diff(np.log(tau_vec)))

    # if they are, we apply the toeplitz trick
    toeplitz_trick = std_diff_freq / mean_diff_freq < 0.01

    if toeplitz_trick and rbf_type != 'PWL':  # apply the toeplitz trick to compute the M matrix

        R = np.array([inner_prod_rbf_2(freq_vec[0], freq_vec[n], epsilon, rbf_type) for n in range(N_taus)])
        C = np.array([inner_prod_rbf_2(freq_vec[n], freq_vec[0], epsilon, rbf_type) for n in range(N_taus)])

        out_M = toeplitz(C, R)

    elif rbf_type == 'PWL':

        out_L_temp = np.zeros((N_taus - 2, N_taus))

        for p in range(N_taus - 2):
            delta_loc = log(tau_vec[p + 1] / tau_vec[p])

            if p == 0 or p == N_taus - 3:
                out_L_temp[p, p] = 2. / (delta_loc ** 2)
                out_L_temp[p, p + 1] = -4. / (delta_loc ** 2)
                out_L_temp[p, p + 2] = 2. / (delta_loc ** 2)
            else:
                out_L_temp[p, p] = 1. / (delta_loc ** 2)
                out_L_temp[p, p + 1] = -2. / (delta_loc ** 2)
                out_L_temp[p, p + 2] = 1. / (delta_loc ** 2)

        out_M = out_L_temp.T @ out_L_temp

    else:  # compute rbf with brute force

        for n in range(N_taus):
            for m in range(N_taus):
                out_M[n, m] = inner_prod_rbf_2(freq_vec[n], freq_vec[m], epsilon, rbf_type)

    return out_M


def quad_format_separate(A, b, M, lambda_value):
    
    """
       This function reformats the DRT regression problem
       (using either the real or imaginary part of the impedance)
       as a quadratic program as follows:
                min (x^T*H*x + c^T*x) 
                under the constraint that x => 0 
                where H = 2*(A^T*A + lambda_value*M) and c = -2*b^T*A        
       Inputs: 
            A: discretization matrix
            b: vector of the real or imaginary part of the impedance
            M: differentiation matrix
            lambda_value: regularization parameter used in Tikhonov regularization
       Outputs: 
            matrix H
            vector c
    """
    
    H = 2*(A.T@A+lambda_value*M)
    H = (H.T+H)/2
    c = -2*b.T@A
    
    return H, c


def quad_format_combined(A_re, A_im, Z_re, Z_im, M, lambda_value):
    
    """
       This function reformats the DRT regression 
       (using both real and imaginary parts of the impedance)
       as a quadratic program 
       
       Inputs:
            A_re: discretization matrix for the real part of the impedance
            A_im: discretization matrix for the imaginary part of the impedance
            Z_re: vector of the real parts of the impedance
            Z_im: vector of the imaginary parts of the impedance
            M: differentiation matrix
            lambda_value: regularization parameter used in Tikhonov regularization

       Outputs: 
            Matrix H
            Vector c
    """
    
    H = 2*((A_re.T@A_re+A_im.T@A_im)+lambda_value*M)
    H = (H.T+H)/2
    c = -2*(Z_im.T@A_im+Z_re.T@A_re)

    return H, c


def solve_gamma(A_re, A_im, Z_re, Z_im, M, lambda_value):
    
    """
    This function solves a quadratic programming problem using the CVXOPT library.

    Inputs:
    - A_re, A_im: Real and imaginary parts of the discretization matrix A
    - Z_re, Z_im: Real and imaginary parts of the impedance Z
    - M: the derivatives matrix M
    - lambda_value: Regularization parameter

    Returns:
    - x: DRT vector
    """

    ## bound matrix
    # G@x<=0
    G = matrix(-np.identity(A_re.shape[0]))
    h = matrix(np.zeros(A_re.shape[0]))
    # Formulate the quadratic programming problem
    H, c = quad_format_combined(A_re, A_im, Z_re, Z_im, M, lambda_value)
    # Solve the quadratic programming problem
    sol = solvers.qp(matrix(H), matrix(c), G, h)
    x = np.array(sol['x']).flatten()

    return x


def optimal_lambda(A_re, A_im, Z_re, Z_im, M, data_used, induct_used, log_lambda_0, cv_type):
    
    """
    This function returns the optimized regularization parameter using various cross-validation methods.
    Inputs:
        A_re: discretization matrix for the real part of the impedance
        A_im: discretization matrix for the imaginary part of the impedance
        Z_re: vector of the real parts of the impedance
        Z_im: vector of the imaginary parts of the impedance
        M: derivative matrix
        data_used: part of the EIS spectrum used for regularization
        induct_used: treatment of the inductance part
        log_lambda_0: initial guess for the regularization parameter
        cv_type: cross-validation method
    Output:
        Optimized regularization parameter based on the any chosen cross-validation method
    """
    
    # interval of values for the regularization parameter
    bnds = [(log(10**-7),log(10**0))] 
    
    # Generalized cross-validation (GCV) method
    if cv_type == 'GCV': 
        res = minimize(param.compute_GCV, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M, data_used, induct_used), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('GCV')
    
    # Modified generalized cross-validation (mGCV) method
    elif cv_type == 'mGCV': 
        res = minimize(param.compute_mGCV, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M, data_used, induct_used), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('mGCV')
        
    # Robust generalized cross-validation (rGCV) method 
    elif cv_type == 'rGCV': 
        res = minimize(param.compute_rGCV, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M, data_used, induct_used), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('rGCV')  
    
    # L-curve (LC) method
    elif cv_type == 'LC':
        res = minimize(param.compute_LC, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M, data_used, induct_used), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('LC')
    
    # real-imaginary (re-im) discrepancy method
    elif cv_type == 're-im':
        res = minimize(param.compute_re_im_cv, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M, data_used, induct_used), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('re-im')
        
    # k-fold cross-validation (kf) method
    elif cv_type == 'kf':  
        res = minimize(param.compute_kf_cv, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M, data_used, induct_used), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('kf') 

    lambda_value = exp(res.x)

    return lambda_value


def pretty_plot(width=8, height=None, plt=None, dpi=None, color_cycle=("qualitative", "Set1_9")):
    
    """
       This function provides a publication-quality plot.
    
       Inputs:
           width (float): Width of plot in inches. Defaults to 8in.
           height (float): Height of plot in inches. Defaults to width * golden ratio.
           plt (matplotlib.pyplot): If plt is supplied, changes will be made to an existing plot. Otherwise, a new plot will be created.
           dpi (int): Sets dot per inch for figure. Defaults to 300.
           color_cycle (tuple): Set the color cycle for new plots to one of the color sets in palettable. Defaults to a qualitative Set1_9.
       Outputs:
           Matplotlib plot object with properly sized fonts.
    """
    
    ticksize = int(width * 2.5)

    golden_ratio = (sqrt(5) - 1) / 2

    if not height:
        height = int(width * golden_ratio)

    if plt is None:
        
        #import matplotlib.pyplot as plt
        #import importlib
        #mod = importlib.import_module("palettable.colorbrewer.%s" % color_cycle[0])
        #colors = getattr(mod, color_cycle[1]).mpl_colors
        # from cycler import cycler

        plt.figure(figsize=(width, height), facecolor="w", dpi=dpi)
        ax = plt.gca()
        # ax.set_prop_cycle(cycler('color', colors))
        
    else:
        
        fig = plt.gcf()
        fig.set_size_inches(width, height)
        
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    
    # shade with appropriate colors
    ax = plt.gca()
    ax.set_title(ax.get_title(), size=width * 4)

    labelsize = int(width * 3)

    ax.set_xlabel(ax.get_xlabel(), size=labelsize)
    ax.set_ylabel(ax.get_ylabel(), size=labelsize)

    return plt