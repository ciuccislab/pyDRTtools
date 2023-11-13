# -*- coding: utf-8 -*-
__authors__ = 'Francesco Ciucci, Ting Hei Wan, Baptiste Py, Adeleke Maradesa'

__date__ = '13th November, 2023'


import numpy as np
from numpy import exp
from math import pi, log, sqrt
from scipy import integrate
from scipy.optimize import fsolve, minimize
from sklearn.model_selection import KFold
from numpy.linalg import norm, cholesky
from scipy.linalg import toeplitz
# import cvxpy as cp
import cvxopt
from numpy import *


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
       This function generates the elements of A_re for the radial-basis-function (RBF) expansion
       Inputs: 
            freq_n: frequency
            tau_m: log timescale (log(1/freq_m))
            epsilon : shape factor of radial basis functions used for discretization
            rbf_type: selected RBF type
       Outputs:
            Elements of A_re for the RBF expansion
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
       This function generates the elements of A_im for RBF expansion
       Inputs:
           freq_n :frequency
           tau_m : log timescale (log(1/freq_m))
           epsilon  : shape factor of radial basis functions used for discretization
           rbf_type : selected RBF type    
       Outputs:
           Elements of A_im for the RBF expansion
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
       This function is used to compute epsilon, i.e., the shape factor of the radial basis functions used for discretization. 
       Inputs:
            freq: frequency
            coeff: scalar such that the full width at half maximum (FWHM) of the RBF is equal to 1/coeff times the average relaxation time spacing in logarithm scale
            rbf_type: selected RBF type 
            shape_control: shape of the RBF, which is set with either the coefficient, or with the option "shape factor" through the shape factor 𝜇
       Output: 
           epsilon (shape factor of radial basis functions used for discretization)
    """ 
    
    N_freq = freq.shape[0]
    
    if rbf_type == 'Piecewise Linear':
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
       This function computes the inner product of the first derivatives of the RBFs with respect to tau_n=log(1/freq_n) and tau_m = log(1/freq_m)
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
       This function computes the inner product of the second derivatives of the RBFs with respect to tau_n=log(1/freq_n) and tau_m = log(1/freq_m)
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
    
    if rbf_type == 'Piecewise Linear':
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
    
    if rbf_type == 'Piecewise Linear':
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


def assemble_A_re(freq_vec, tau_vec, epsilon, rbf_type, flag1='simple', flag2='impedance'):
    
    """
       This function computes the discretization matrix, A_re, for the real part of the impedance
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
    omega_vec = 2.*pi*freq_vec
    N_freqs = freq_vec.size
    N_taus = tau_vec.size
    
    if flag1 == 'simple':
        
        # define the A_re output matrix
        out_A_re = np.zeros((N_freqs, N_taus))
    
        # check if the frequencies are sufficiently log spaced
        std_diff_freq = np.std(np.diff(np.log(1/freq_vec)))
        mean_diff_freq = np.mean(np.diff(np.log(1/freq_vec)))
    
        # check if the frequencies are sufficiently log spaced and that N_freqs = N_taus
        toeplitz_trick = std_diff_freq/mean_diff_freq<0.01 and N_freqs == N_taus 

        if toeplitz_trick and rbf_type != 'Piecewise Linear': # use toeplitz trick
            
            R = np.zeros(N_taus)
            C = np.zeros(N_freqs)
        
            for p in range(0, N_freqs):
            
                C[p] = g_i(freq_vec[p], tau_vec[0], epsilon, rbf_type)
        
            for q in range(0, N_taus):
            
                R[q] = g_i(freq_vec[0], tau_vec[q], epsilon, rbf_type)        
                        
            out_A_re= toeplitz(C,R) 

        else: # use brute force
            
            for p in range(0, N_freqs):
                for q in range(0, N_taus):
            
                    if rbf_type == 'Piecewise Linear':  # see (A.3a) and (A.4) in [2]              
                        if q == 0:
                            out_A_re[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                        elif q == N_taus-1:
                            out_A_re[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                        else:
                            out_A_re[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])                    
                
                    else:
                        out_A_re[p, q]= g_i(freq_vec[p], tau_vec[q], epsilon, rbf_type)
    
    else: # BHT run
    
        out_A_re = np.zeros((N_freqs, N_taus+1))
        out_A_re[:,0] = 1.
        
        if flag2 == 'impedance': # for the impedance calculations
        
            for p in range(0, N_freqs):
                for q in range(0, N_taus): # see (11a) in [2]
                    if q == 0:
                        out_A_re[p, q+1] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                    elif q == N_taus-1:
                        out_A_re[p, q+1] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                    else:
                        out_A_re[p, q+1] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])
        
        else: # for the admittance calculations
        
            for p in range(0, N_freqs):
                for q in range(0, N_taus): # see (16a) in the supplementary information (SI) of [2]
                    if q == 0:
                        out_A_re[p, q+1] = 0.5*(omega_vec[p]**2*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                    elif q == N_taus-1:
                        out_A_re[p, q+1] = 0.5*(omega_vec[p]**2*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                    else:
                        out_A_re[p, q+1] = 0.5*(omega_vec[p]**2*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])

    return out_A_re


def assemble_A_im(freq_vec, tau_vec, epsilon, rbf_type, flag1='simple', flag2='impedance'):
    
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
    omega_vec = 2.*pi*freq_vec
    N_freqs = freq_vec.size
    N_taus = tau_vec.size
    
    if flag1 == 'simple':

        # define the A_re output matrix
        out_A_im = np.zeros((N_freqs, N_taus))
    
        # check if the frequencies are sufficiently log spaced
        std_diff_freq = np.std(np.diff(np.log(1/freq_vec)))
        mean_diff_freq = np.mean(np.diff(np.log(1/freq_vec)))
    
        # check if the frequencies are sufficiently log spaced and that N_freqs = N_taus
        toeplitz_trick = std_diff_freq/mean_diff_freq<0.01 and N_freqs == N_taus 
    
        if toeplitz_trick and rbf_type != 'Piecewise Linear': # use toeplitz trick
        
            R = np.zeros(N_taus)
            C = np.zeros(N_freqs)
        
            for p in range(0, N_freqs):
            
                C[p] = - g_ii(freq_vec[p], tau_vec[0], epsilon, rbf_type)
        
            for q in range(0, N_taus):
            
                R[q] = - g_ii(freq_vec[0], tau_vec[q], epsilon, rbf_type)        
                        
            out_A_im = toeplitz(C,R) 

        else: # use brute force
        
            for p in range(0, N_freqs):
                for q in range(0, N_taus):
            
                    if rbf_type == 'Piecewise Linear': # see (A.3b) and (A.5) in [2]               
                        if q == 0:
                            out_A_im[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                        elif q == N_taus-1:
                            out_A_im[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                        else:
                            out_A_im[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])                    
                
                    else:
                        out_A_im[p, q]= - g_ii(freq_vec[p], tau_vec[q], epsilon, rbf_type)
    
    else: # BHT run
    
        out_A_im = np.zeros((N_freqs, N_taus+1))
        out_A_im[:,0] = omega_vec

        if flag2 == 'impedance': # for the impedance calculations
        
            for p in range(0, N_freqs):
                for q in range(0, N_taus): # see (11b) in [2]
                    if q == 0:
                        out_A_im[p, q+1] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                    elif q == N_taus-1:
                        out_A_im[p, q+1] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                    else:
                        out_A_im[p, q+1] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])
        
        else:  # for the admittance calculations
        
            for p in range(0, N_freqs):
                for q in range(0, N_taus): # see (16b) in the SI of [2]
                    if q == 0:
                        out_A_im[p, q+1] = 0.5*(omega_vec[p])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                    elif q == N_taus-1:
                        out_A_im[p, q+1] = 0.5*(omega_vec[p])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                    else:
                        out_A_im[p, q+1] = 0.5*(omega_vec[p])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])
 
    return out_A_im


def assemble_M_1(tau_vec, epsilon, rbf_type, flag='simple'): # see (38) in [1]
    
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
    
    freq_vec = 1/tau_vec 
    
    # first get the number of collocation points
    N_taus = tau_vec.size
    N_freq = freq_vec.size
    
    if flag == 'simple': # simple run
    
        # define the M output matrix
        out_M = np.zeros([N_taus, N_taus])
    
        # check if the collocation points are sufficiently log spaced
        std_diff_freq = np.std(np.diff(np.log(tau_vec)));
        mean_diff_freq = np.mean(np.diff(np.log(tau_vec)));
    
        # if they are, we apply the toeplitz trick  
        toeplitz_trick = std_diff_freq/mean_diff_freq<0.01
    
        if toeplitz_trick and rbf_type != 'Piecewise Linear': # apply the toeplitz trick to compute the M matrix 
        
            R = np.zeros(N_taus)
            C = np.zeros(N_taus)
        
            for n in range(0,N_taus):
                C[n] = inner_prod_rbf_1(freq_vec[0], freq_vec[n], epsilon, rbf_type)
            
            for m in range(0,N_taus):
                R[m] = inner_prod_rbf_1(freq_vec[m], freq_vec[0], epsilon, rbf_type)    
        
            out_M = toeplitz(C,R) 
         
        elif rbf_type == 'Piecewise Linear': 
       
            out_L_temp = np.zeros([N_freq-1, N_freq])
        
            for iter_freq_n in range(0,N_freq-1):
                delta_loc = log((1/freq_vec[iter_freq_n+1])/(1/freq_vec[iter_freq_n]))
                out_L_temp[iter_freq_n,iter_freq_n] = -1/delta_loc
                out_L_temp[iter_freq_n,iter_freq_n+1] = 1/delta_loc

            out_M = out_L_temp.T@out_L_temp
    
        else: # compute rbf with brute force
    
            for n in range(0, N_taus):
                for m in range(0, N_taus):            
                    out_M[n,m] = inner_prod_rbf_1(freq_vec[n], freq_vec[m], epsilon, rbf_type)
    
    else: # BHT run ; see (18) in [3]
        
        out_M = np.zeros((N_taus-2, N_taus+1))
        
        for p in range(0, N_taus-2):

            delta_loc = log(tau_vec[p+1]/tau_vec[p])
            
            if p==0:
                out_M[p,p+1] = -3./(2*delta_loc)
                out_M[p,p+2] = 4./(2*delta_loc)
                out_M[p,p+3] = -1./(2*delta_loc)
                
            elif p == N_taus-2:
                out_M[p,p]   = 1./(2*delta_loc)
                out_M[p,p+1] = -4./(2*delta_loc)
                out_M[p,p+2] = 3./(2*delta_loc)
                
            else:
                out_M[p,p] = 1./(2*delta_loc)
                out_M[p,p+2] = -1./(2*delta_loc)
        
    return out_M


def assemble_M_2(tau_vec, epsilon, rbf_type, flag='simple'): # see (38) in [1]
    
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
    
    freq_vec = 1/tau_vec            
    
    # first get number of collocation points
    N_taus = tau_vec.size
    
    if flag == 'simple': # simple run
    
        # define the M output matrix
        out_M = np.zeros([N_taus, N_taus])
    
        # check if the collocation points are sufficiently log spaced
        std_diff_freq = np.std(np.diff(np.log(tau_vec)));
        mean_diff_freq = np.mean(np.diff(np.log(tau_vec)));
    
        # if they are, we apply the toeplitz trick  
        toeplitz_trick = std_diff_freq/mean_diff_freq<0.01
    
        if toeplitz_trick and rbf_type != 'Piecewise Linear': # apply the toeplitz trick to compute the M matrix 
        
            R = np.zeros(N_taus)
            C = np.zeros(N_taus)
        
            for n in range(0,N_taus):
                C[n] = inner_prod_rbf_2(freq_vec[0], freq_vec[n], epsilon, rbf_type) # later, we shall use tau instead of freq
            
            for m in range(0,N_taus):
                R[m] = inner_prod_rbf_2(freq_vec[m], freq_vec[0], epsilon, rbf_type) # later, we shall use tau instead of freq
        
            out_M = toeplitz(C,R) 
         
        elif rbf_type == 'Piecewise Linear':
        
            out_L_temp = np.zeros((N_taus-2, N_taus))
    
            for p in range(0, N_taus-2):
                delta_loc = log(tau_vec[p+1]/tau_vec[p])
            
                if p == 0 or p == N_taus-3:
                    out_L_temp[p,p] = 2./(delta_loc**2)
                    out_L_temp[p,p+1] = -4./(delta_loc**2)
                    out_L_temp[p,p+2] = 2./(delta_loc**2)
                    
                else:
                    out_L_temp[p,p] = 1./(delta_loc**2)
                    out_L_temp[p,p+1] = -2./(delta_loc**2)
                    out_L_temp[p,p+2] = 1./(delta_loc**2)
                
            out_M = out_L_temp.T@out_L_temp
    
        else: # compute rbf with brute force
    
            for n in range(0, N_taus):
                for m in range(0, N_taus):            
                    out_M[n,m] = inner_prod_rbf_2(freq_vec[n], freq_vec[m], epsilon, rbf_type)
                    
    else: # BHT run
        
        out_M = np.zeros((N_taus-2, N_taus+1))
        
        for p in range(0, N_taus-2):

            delta_loc = log(tau_vec[p+1]/tau_vec[p])
            
            if p==0 or p == N_taus-3:
                out_M[p,p+1] = 2./(delta_loc**2)
                out_M[p,p+2] = -4./(delta_loc**2)
                out_M[p,p+3] = 2./(delta_loc**2)
                
            else:
                out_M[p,p+1] = 1./(delta_loc**2)
                out_M[p,p+2] = -2./(delta_loc**2)
                out_M[p,p+3] = 1./(delta_loc**2)
        
    return out_M


def quad_format_separate(A, b, M, lambda_value):
    
    """
       This function reformats the DRT regression as a quadratic program using either the real or imaginary part of the impedance as follows:
                min (x^T*H*x + c^T*x) under the constraint that x => 0 with H = 2*(A^T*A + lambda_value*M) and c = -2*b^T*A        
       Inputs: 
            A: discretization matrix
            b: vector of the real or imaginary part of the impedance
            M: differentiation matrix
            lambda_value: regularization parameter used in Tikhonov regularization
       Outputs: 
            Matrix H
            Vector c
    """
    
    H = 2*(A.T@A+lambda_value*M)
    H = (H.T+H)/2
    c = -2*b.T@A
    
    return H, c


def quad_format_combined(A_re, A_im, Z_re, Z_im, M, lambda_value):
    
    """
       This function reformats the DRT regression as a quadratic program using both real and imaginary parts of the impedance
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


# def cvxpy_solve_qp(H, c):

#     """ 
#        This function uses cvxpy to minimize the quadratic problem 0.5*x^T*H*x + c^T*x under the non-negativity constraint.
#        Inputs: 
#            H: matrix
#            c: vector
#         Output: 
#            Vector solution of the aforementioned problem
#     """
    
#     N_out = c.shape[0]
#     x = cp.Variable(shape = N_out, value = np.ones(N_out))
#     h = np.zeros(N_out)
    
#     prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, H) + c@x), [x >= h])
#     prob.solve(verbose = True, eps_abs = 1E-10, eps_rel = 1E-10, sigma = 1.00e-08, 
#                max_iter = 200000, eps_prim_inf = 1E-5, eps_dual_inf = 1E-5)

#     gamma = x.value
    
#     return gamma


def cvxopt_solve_qpr(P, q, G=None, h=None, A=None, b=None):
    
    """
       This function uses cvxopt to minimize the quadratic problem 0.5*x^T*P*x + q^T*x under the constraints that G*x <= h (element-wise) and A*x = b. 
       Inputs: 
           P: matrix
           q: vector
           G: matrix
           h: vector
           A: matrix
           B: vector
       Output: 
           Vector solution of the aforementioned poblem
    """
    
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    
    if G is not None: # in case the element-wise inequality constraint G*x <= b is included
    
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        
    if A is not None: # in case the equality constraint A*x = b is included
    
         args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
        
    cvxopt.solvers.options['abstol'] = 1e-15
    cvxopt.solvers.options['reltol'] = 1e-15 ## could be 1e-15
    sol = cvxopt.solvers.qp(*args)
    
    if 'optimal' not in sol['status']:
        
        return None
    
    return np.array(sol['x']).reshape((P.shape[1],))


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
        
        import matplotlib.pyplot as plt
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


# Part 2: Selection of the regularization parameter for ridge regression

def is_PD(A):
    
    """
       This function checks if a matrix A is positive-definite using Cholesky transform
    """
    
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

    
def nearest_PD(A):
    
    """
       This function finds the nearest positive definite matrix of a matrix A. The code is based on John D'Errico's "nearestSPD" code on Matlab [1]. More details can be found in the following two references:
         https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
         N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    
    B = (A + A.T)/2
    _, Sigma_mat, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(Sigma_mat), V))

    A_nPD = (B + H) / 2
    A_symm = (A_nPD + A_nPD.T) / 2

    k = 1
    I = np.eye(A_symm.shape[0])

    while not is_PD(A_symm): # the Matlab function chol accepts matrices with eigenvalue = 0, but numpy does not so we replace the Matlab function eps(min_eig) with the following one
        
        eps = np.spacing(np.linalg.norm(A_symm))
        min_eig = min(0, np.min(np.real(np.linalg.eigvals(A_symm))))
        A_symm += I * (-min_eig * k**2 + eps)
        k += 1

    return A_symm


def compute_GCV(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
       This function computes the score for the generalized cross-validation (GCV) approach.
       Reference: G. Wahba, A comparison of GCV and GML for choosing the smoothing parameter in the generalized spline smoothing problem, Ann. Statist. 13 (1985) 1378–1402.
       Inputs: 
           log_lambda: regularization parameter
           A_re: discretization matrix for the real part of the impedance
           A_im: discretization matrix for the real part of the impedance
           Z_re: vector of the real parts of the impedance
           Z_im: vector of the imaginary parts of the impedance
           M: differentiation matrix 
       Output:
           GCV score
    """
    
    lambda_value = exp(log_lambda)
    
    A = np.concatenate((A_re, A_im), axis = 0) # matrix A with A_re and A_im ; see (5) in [4]
    Z = np.concatenate((Z_re, Z_im), axis = 0) # stacked impedance
    
    n_cv = Z.shape[0] # n_cv = 2*N_freqs with N_freqs the number of EIS frequencies
    
    A_agm = A.T@A + lambda_value*M # see (13) in [4]
    
    if (is_PD(A_agm)==False): # check if A_agm is positive-definite
        A_agm = nearest_PD(A_agm) 
        
    L_agm = cholesky(A_agm) # Cholesky transform to inverse A_agm
    inv_L_agm = np.linalg.inv(L_agm)
    inv_A_agm = inv_L_agm.T@inv_L_agm # inverse of A_agm
    A_GCV = A@inv_A_agm@A.T  # see (13) in [4]
    
    # GCV score; see (13) in [4]
    GCV_num = 1/n_cv*norm((np.eye(n_cv)-A_GCV)@Z)**2 # numerator
    GCV_dom = (1/n_cv*np.trace(np.eye(n_cv)-A_GCV))**2 # denominator
    
    GCV_score = GCV_num/GCV_dom
    
    return GCV_score


def compute_mGCV(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
       This function computes the score for the modified generalized cross validation (mGCV) approach.
       Reference: Y.J. Kim, C. Gu, Smoothing spline Gaussian regression: More scalable computation via efficient approximation, J. Royal Statist. Soc. 66 (2004) 337–356.
       Inputs: 
           log_lambda: regularization parameter
           A_re: discretization matrix for the real part of the impedance
           A_im: discretization matrix for the real part of the impedance
           Z_re: vector of the real parts of the impedance
           Z_im: vector of the imaginary parts of the impedance
           M: differentiation matrix 
       Output:
           mGCV score
    """
    
    lambda_value = exp(log_lambda)
    
    A = np.concatenate((A_re, A_im), axis = 0) # see (5) in [4]
    Z = np.concatenate((Z_re, Z_im), axis = 0)
    
    n_cv = Z.shape[0] # 2*number of frequencies
    
    A_agm = A.T@A + lambda_value*M # see (13) in [4]

    if (is_PD(A_agm)==False):
        A_agm = nearest_PD(A_agm)
    
    L_agm = cholesky(A_agm) # Cholesky transform to inverse A_agm
    inv_L_agm = np.linalg.inv(L_agm)
    inv_A_agm = inv_L_agm.T@inv_L_agm # inverse of A_agm
    A_GCV = A@inv_A_agm@A.T # see (13) in [4]
    
    # the stabilization parameter, rho, is computed as described by Kim et al.
    rho = 2 # see (15) in [4]
    
    # mGCV score ; see (14) in [4]
    mGCV_num = 1/n_cv*norm((np.eye(n_cv)-A_GCV)@Z)**2 # numerator
    mGCV_dom = ((1/n_cv)*(np.trace(np.eye(n_cv)-rho*A_GCV)))**2 # denominator
    mGCV_score = mGCV_num/mGCV_dom
    
    return mGCV_score


def compute_rGCV(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
       This function computes the score for the robust generalized cross-validation (rGCV) approach.
       Reference: M. A. Lukas, F. R. de Hoog, R. S. Anderssen, Practical use of robust GCV and modified GCV for spline smoothing, Comput. Statist. 31 (2016) 269–289.   
       Inputs: 
           log_lambda: regularization parameter
           A_re: discretization matrix for the real part of the impedance
           A_im: discretization matrix for the real part of the impedance
           Z_re: vector of the real parts of the impedance
           Z_im: vector of the imaginary parts of the impedance
           M: differentiation matrix 
       Output:
           rGCV score    
    """
     
    lambda_value = exp(log_lambda)
    
    A = np.concatenate((A_re, A_im), axis = 0) # see (5) in [4]
    Z = np.concatenate((Z_re, Z_im), axis = 0)
    
    n_cv = Z.shape[0] # 2*number of frequencies
    
    A_agm = A.T@A + lambda_value*M # see (13) in [4]

    if (is_PD(A_agm)==False):
        A_agm = nearest_PD(A_agm)
    
    L_agm = cholesky(A_agm) # Cholesky transform to inverse A_agm
    inv_L_agm = np.linalg.inv(L_agm)
    inv_A_agm = inv_L_agm.T@inv_L_agm # inverse of A_agm
    A_GCV = A@inv_A_agm@A.T # see (13) in [4]
    
    # GCV score ; see (13) in [4]
    rGCV_num = 1/n_cv*norm((np.eye(n_cv)-A_GCV)@Z)**2
    rGCV_dom = ((1/n_cv)*(np.trace(np.eye(n_cv)-A_GCV)))**2
    rGCV = rGCV_num/rGCV_dom
    
    # the robust parameter, xsi, is computed as described in Lukas et al.
    xi = 0.3 # see (16) in [4]
    
    # mu_2 parameter ; see (16) in [4]
    mu_2 = (1/n_cv)*np.trace(A_GCV.T@A_GCV)
    
    # rGCV score ; see (16) in [4]
    rGCV_score = (xi + (1-xi)*mu_2)*rGCV
        
    return rGCV_score


def compute_re_im_cv(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
       This function computes the score for real-imaginary discrepancy (re-im).
       Inputs: 
           log_lambda: regularization parameter
           A_re: discretization matrix for the real part of the impedance
           A_im: discretization matrix for the real part of the impedance
           Z_re: vector of the real parts of the impedance
           Z_im: vector of the imaginary parts of the impedance
           M: differentiation matrix 
       Output:
           re-im score
    """
    
    lambda_value = exp(log_lambda)
    
    # non-negativity constraint on the DRT gmma
    lb = np.zeros([Z_re.shape[0]+1]) # + 1 if a resistor or an inductor is included in the DRT model
    bound_mat = np.eye(lb.shape[0]) 
    
    # quadratic programming through cvxopt  quad_format_separate
    H_re, c_re = quad_format_separate(A_re, Z_re, M, lambda_value)
    gamma_ridge_re = cvxopt_solve_qpr(H_re, c_re, -bound_mat, lb)
    H_im, c_im = quad_format_separate(A_im, Z_im, M, lambda_value)
    gamma_ridge_im = cvxopt_solve_qpr(H_im, c_im, -bound_mat, lb)
    
    # stacking the resistance R and inductance L on top of gamma_ridge_im and gamma_ridge_re, repectively
    gamma_ridge_re_cv = np.concatenate((np.array([0, gamma_ridge_re[1]]), gamma_ridge_im[2:]))
    gamma_ridge_im_cv = np.concatenate((np.array([gamma_ridge_im[0], 0]), gamma_ridge_re[2:]))
    
    # re-im score ; see (13) in [2] and (17) in [4]
    re_im_cv_score = norm(Z_re - A_re@gamma_ridge_re_cv)**2+norm(Z_im-A_im@gamma_ridge_im_cv)**2
    
    return re_im_cv_score


def compute_kf_cv(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
       This function computes the k-fold (kf) score.
       Inputs: 
           log_lambda: regularization parameter
           A_re: discretization matrix for the real part of the impedance
           A_im: discretization matrix for the real part of the impedance
           Z_re: vector of the real parts of the impedance
           Z_im: vector of the imaginary parts of the impedance
           M: differentiation matrix 
       Output:
           kf score
    """

    lambda_value = exp(log_lambda)
    
    # non-negativity constraint on the DRT gamma
    lb = np.zeros([Z_re.shape[0]+1])
    bound_mat = np.eye(lb.shape[0])
    
    # parameters for kf
    N_splits = 5 # N_splits=N_freq correspond to leave-one-out cross-validation
    random_state = 34054 + compute_kf_cv.counter*100  # change random state for each experiment
    kf = KFold(n_splits = N_splits, shuffle = True, random_state = random_state)                
    kf_cv = 0
    
    # train and test 
    for train_index, test_index in kf.split(Z_re):
        
        # step 1: preparation of the train and test sets
        print("TRAIN:", train_index, "TEST:", test_index)
        A_re_train, A_re_test = A_re[train_index,:], A_re[test_index,:]
        A_im_train, A_im_test = A_im[train_index,:], A_im[test_index,:]        
        Z_re_train, Z_re_test = Z_re[train_index], Z_re[test_index]
        Z_im_train, Z_im_test = Z_im[train_index], Z_im[test_index]
        
        # step 2: qudratic programming to obtain the DRT
        H_combined, c_combined = quad_format_combined(A_re_train, A_im_train, Z_re_train, Z_im_train, M, lambda_value)
        gamma_ridge = cvxopt_solve_qpr(H_combined, c_combined, -bound_mat, lb)
        
        # step 3: update of the kf scores    
        kf_cv += 1/Z_re_test.shape[0]*(norm(Z_re_test-A_re_test@gamma_ridge)**2 + norm(Z_im_test-A_im_test@gamma_ridge)**2)
    
    # kf score ; see section 1.2 in the SI of [4]
    kf_cv_score = kf_cv/N_splits
    
    return kf_cv_score
compute_kf_cv.counter = 0


def compute_LC(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
       This function computes the score for L curve (LC)
       Reference: P.C. Hansen, D.P. O’Leary, The use of the L-curve in the regularization of discrete ill-posed problems, SIAM J. Sci. Comput. 14 (1993) 1487–1503.
       Inputs: 
           log_lambda: regularization parameter
           A_re: discretization matrix for the real part of the impedance
           A_im: discretization matrix for the real part of the impedance
           Z_re: vector of the real parts of the impedance
           Z_im: vector of the imaginary parts of the impedance
           M: differentiation matrix 
       Output:
           LC score
    """
    
    lambda_value = exp(log_lambda)
    
    A = np.concatenate((A_re, A_im), axis = 0) # matrix A with A_re and A_im; # see (5) in [4]
    Z = np.concatenate((Z_re, Z_im), axis = 0) # stacked impedance
    
    # numerator eta_num of the first derivative of eta = log(||Z_exp - Ax||^2)
    A_agm = A.T@A + lambda_value*M # see (13) in [4]
    if (is_PD(A_agm)==False):
        A_agm = nearest_PD(A_agm)
           
    L_agm = cholesky(A_agm) # Cholesky transform to inverse A_agm
    inv_L_agm = np.linalg.inv(L_agm)
    inv_A_agm = inv_L_agm.T@inv_L_agm # inverse of A_agm
    A_LC = A@((inv_A_agm.T@inv_A_agm)@inv_A_agm)@A.T
    eta_num = Z.T@A_LC@Z 

    # denominator eta_denom of the first derivative of eta
    A_agm_d = A@A.T + lambda_value*np.eye(A.shape[0])
    if (is_PD(A_agm_d)==False):
        A_agm_d = nearest_PD(A_agm_d)
    
    L_agm_d = cholesky(A_agm_d) # Cholesky transform to inverse A_agm_d
    inv_L_agm_d = np.linalg.inv(L_agm_d)
    inv_A_agm_d = inv_L_agm_d.T@inv_L_agm_d
    eta_denom = lambda_value*Z.T@(inv_A_agm_d.T@inv_A_agm_d)@Z
    
    # derivative of eta
    eta_prime = eta_num/eta_denom
    
    # numerator theta_num of the first derivative of theta = log(lambda*||Lx||^2)
    theta_num  = eta_num
    
    # denominator theta_denom of the first derivative of theta
    A_LC_d = A@(inv_A_agm.T@inv_A_agm)@A.T
    theta_denom = Z.T@A_LC_d@Z
    
    # derivative of theta 
    theta_prime = -(theta_num)/theta_denom
    
    # numerator LC_num of the LC score in (19) in [4]
    a_sq = (eta_num/(eta_denom*theta_denom))**2
    p = (Z.T@(inv_A_agm_d.T@inv_A_agm_d)@Z)*theta_denom
    m = (2*lambda_value*Z.T@((inv_A_agm_d.T@inv_A_agm_d)@inv_A_agm_d)@Z)*theta_denom
    q = (2*lambda_value*Z.T@(inv_A_agm_d.T@inv_A_agm_d)@Z)*eta_num 
    LC_num = a_sq*(p+m-q)

    # denominator LC_denom of the LC score
    LC_denom = ((eta_prime)**2 + (theta_prime)**2)**(3/2)
    
    # LC score ; see (19) in [4]
    LC_score = LC_num/LC_denom
    
    return -LC_score 


def optimal_lambda(A_re, A_im, Z_re, Z_im, M, log_lambda_0, cv_type):
    
    """
       This function returns the regularization parameter given an initial guess and a regularization method. For constrained minimization, we use the scipy function sequential least squares programming (SLSQP).
       Inputs: 
           A_re: discretization matrix for the real part of the impedance
           A_im: discretization matrix for the real part of the impedance
           Z_re: vector of the real parts of the impedance
           Z_im: vector of the imaginary parts of the impedance
           M: differentiation matrix 
           log_lambda0: initial guess for the regularization parameter
           cv_type: regularization method
       Output:
           optimized regularization parameter given the regularization method chosen
    """
    
    # credible for the lambda values
    bnds = [(log(10**-7),log(10**0))] 
    
    # GCV method
    if cv_type == 'GCV': 
        res = minimize(compute_GCV, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('GCV')
    
    # mGCV method
    elif cv_type == 'mGCV': 
        res = minimize(compute_mGCV, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('mGCV')
        
    # rGCV method
    elif cv_type == 'rGCV': 
        res = minimize(compute_rGCV, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('rGCV')  
    
    # L-curve method
    elif cv_type == 'LC':
        res = minimize(compute_LC, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('LC')
    
    # re-im discrepancy
    elif cv_type == 're-im':
        res = minimize(compute_re_im_cv, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('re-im')
        
    # k-fold 
    else: #elif cv_type == 'kf':  
        res = minimize(compute_kf_cv, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('kf') 
    
    lambda_value = exp(res.x)
    
    return lambda_value

# Part 3: Peak analysis

def gauss_fct(p, tau, N_peaks): # N_peaks is the number of peaks in the DRT spectrum
    
    gamma_out = np.zeros_like(tau) # sum of Gaussian functions, whose parameters (the prefactor sigma_f, mean mu_log_tau, and standard deviation 1/inv_sigma for each DRT peak) are encapsulated in p
    
    for k in range(N_peaks):
        
        sigma_f, mu_log_tau, inv_sigma = p[3*k:3*k+3] 
        gaussian_out = sigma_f**2*np.exp(-inv_sigma**2/2*((np.log(tau) - mu_log_tau)**2)) # we use inv_sigma because this leads to less computational problems (no exploding gradient when sigma->0)
        gamma_out += gaussian_out 
    return gamma_out    