import numpy as np
from numpy import exp
from math import pi, log, sqrt
from scipy import integrate
from scipy.optimize import fsolve
from scipy.linalg import toeplitz
import cvxpy as cp
from numpy.linalg import norm, cholesky
from math import pi, log, sqrt, log10
from scipy.optimize import fsolve, minimize, LinearConstraint, Bounds
from scipy.linalg import toeplitz, hankel
from sklearn.model_selection import KFold
from numpy import linalg as la
from numpy import *
import cvxopt


"""
This file contains the main functions related to DRTtools to carry out DRT deconvolution and optimally select the regularization parameter for ridge regression.
"""


rbf_switch = {
            'Gaussian': lambda x, epsilon: exp(-(epsilon*x)**2),
            'C0 Matern': lambda x, epsilon: exp(-abs(epsilon*x)),
            'C2 Matern': lambda x, epsilon: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
            'C4 Matern': lambda x, epsilon: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
            'C6 Matern': lambda x, epsilon: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
            'Inverse Quadratic': lambda x, epsilon: 1/(1+(epsilon*x)**2),
            'Inverse Quadric': lambda x, epsilon: 1/sqrt(1+(epsilon*x)**2),
            'Cauchy': lambda x, epsilon: 1/(1+abs(epsilon*x))
            }


def g_re(freq_m, tau_n, expansion_type, epsilon):
    
    """
    this function computes (A_re)_mn for the radial basis functions (RBF) approximation given the following inputs:
        - a frequency freq_m
        - a timescale tau_n
        - RBF type - the different types of RBFs are specified in rbf_switch with the corresponding analytical expression-
        - a shape factor epsilon
    """

    alpha = 2*pi*freq_m*tau_n  
    
    rbf = rbf_switch.get(expansion_type)
    integrand_g_re = lambda x: 1./(1.+(alpha**2)*exp(2.*x))*rbf(x, epsilon)
    out_val = integrate.quad(integrand_g_re, -50, 50, epsabs=1E-9, epsrel=1E-9)

    return out_val[0]


def g_im(freq_m, tau_n, expansion_type, epsilon):
    
    """
    this function computes (A_im)_mn for the RBF approximation given the following inputs:
        - a frequency freq_m
        - a timescale tau_n
        - a RBF type
        - a shape factor
    """

    alpha = 2*pi*freq_m*tau_n 

    rbf = rbf_switch.get(expansion_type)
    integrand_g_im = lambda x: alpha/(1./exp(x)+(alpha**2)*exp(x))*rbf(x, epsilon)
    out_val = integrate.quad(integrand_g_im, -50, 50, epsabs=1E-9, epsrel=1E-9)
    
    return out_val[0]


def compute_epsilon(tau_vec, coeff, expansion_type, shape_control):
    
    """
    this function computes the shape factor, i.e. epsilon, of the RBF given the full-width at half maximum (FWHM)
    """

    N_tau = tau_vec.shape[0]

    rbf_switch_0 = {
                    'Gaussian': lambda x: exp(-(x)**2)-0.5,
                    'C0 Matern': lambda x: exp(-abs(x))-0.5,
                    'C2 Matern': lambda x: exp(-abs(x))*(1+abs(x))-0.5,
                    'C4 Matern': lambda x: 1/3*exp(-abs(x))*(3+3*abs(x)+abs(x)**2)-0.5,
                    'C6 Matern': lambda x: 1/15*exp(-abs(x))*(15+15*abs(x)+6*abs(x)**2+abs(x)**3)-0.5,
                    'Inverse Quadratic': lambda x: 1/(1+(x)**2)-0.5,
                    'Inverse Quadric': lambda x: 1/sqrt(1+(x)**2)-0.5,
                    'Cauchy': lambda x: 1/(1+abs(x))-0.5
                    }

    if expansion_type == 'PWL':
        epsilon = [0]
    
    else:
        rbf = rbf_switch_0.get(expansion_type)
    
        if shape_control == 'FWHM': # equivalent to the 'FWHM Coefficient' option in the Matlab DRTtools
            FWHM_coeff = 2*fsolve(rbf, 1)
            delta = abs(np.mean(np.diff(np.log(tau_vec.reshape(N_tau)))))
            epsilon = coeff*FWHM_coeff/delta

        else: # equivalent to the 'shape factor' option in the Matlab DRTtools
            epsilon = coeff
    
    return epsilon[0]
 
    
def inner_prod_rbf_D1(tau_n, tau_m, expansion_type, epsilon):
    
    """
    this function computes the inner product of the first derivative of a given RBF with respect to two timescales tau_n and tau_m
    """
    
    a = -epsilon*log(tau_n/tau_m)
    
    if expansion_type == 'Inverse Quadric':
        y_n = log(tau_n)
        y_m = log(tau_m)

        # could only find numerical version
        rbf_n = lambda y: 1/sqrt(1+(epsilon*(y-y_n))**2)
        rbf_m = lambda y: 1/sqrt(1+(epsilon*(y-y_m))**2)

        # computation of the derivative
        delta = 1E-8
        sqr_drbf_dy = lambda y: 1/(2*delta)*(rbf_n(y+delta)-rbf_n(y-delta))*1/(2*delta)*(rbf_m(y+delta)-rbf_m(y-delta))
        out_val = integrate.quad(sqr_drbf_dy, -50, 50, epsabs=1E-9, epsrel=1E-9)
        out_val = out_val[0]
        
    elif expansion_type == 'Cauchy':
        if a == 0:
            out_val = 2/3*epsilon
            
        else:
            num = abs(a)*(2+abs(a))*(4+3*abs(a)*(2+abs(a)))-2*(1+abs(a))**2*(4+abs(a)*(2+abs(a)))*log(1+abs(a))
            den = abs(a)**3*(1+abs(a))*(2+abs(a))**3
            out_val = 4*epsilon*num/den
        
    else:
        rbf_D1_switch = {
                    'Gaussian': -epsilon*(-1+a**2)*exp(-(a**2/2))*sqrt(pi/2),
                    'C0 Matern': epsilon*(1-abs(a))*exp(-abs(a)),
                    'C2 Matern': epsilon/6*(3+3*abs(a)-abs(a)**3)*exp(-abs(a)),
                    'C4 Matern': epsilon/30*(105+105*abs(a)+30*abs(a)**2-5*abs(a)**3-5*abs(a)**4-abs(a)**5)*exp(-abs(a)),
                    'C6 Matern': epsilon/140*(10395 +10395*abs(a)+3780*abs(a)**2+315*abs(a)**3-210*abs(a)**4-84*abs(a)**5-14*abs(a)**6-abs(a)**7)*exp(-abs(a)),
                    'Inverse Quadratic': 4*epsilon*(4-3*a**2)*pi/((4+a**2)**3)
                    }
        
        out_val = rbf_D1_switch.get(expansion_type)
        
    return out_val


def inner_prod_rbf_D2(tau_n, tau_m, expansion_type, epsilon):
    
    """
    this function computes the inner product of the second derivatives of a given RBF with respect to two timescales tau_n and tau_m
    """
    
    a = epsilon*log(tau_n/tau_m)
    
    if expansion_type == 'Inverse Quadric':
        y_n = log(tau_n)
        y_m = log(tau_m)
           
        sqr_drbf_dy = lambda y: epsilon**2*(y-y_n)/(1+epsilon*(y-y_n)**2)**1.5 * (y-y_m)/(1+epsilon*(y-y_m)**2)**1.5

        out_val = integrate.quad(sqr_drbf_dy, -50, 50, epsabs=1E-9, epsrel=1E-9) # computation of the integral
        out_val = out_val[0]
        
    elif expansion_type == 'Cauchy':
      
        if a == 0:
            out_val = 8/5*epsilon**3
        else:
            num = abs(a)*(2+abs(a))*(-96 +abs(a)*(2+abs(a))*(-30 +abs(a)*(2+abs(a)))*(4+abs(a)*(2+abs(a))))+12*(1+abs(a))**2*(16+abs(a)*(2+abs(a))*(12+abs(a)*(2+abs(a))))*log(1+abs(a))
            den = abs(a)**5*(1+abs(a))*(2+abs(a))**5
            out_val = 8*epsilon**3*num/den
        
    else:
       
        rbf_D2_switch = {
                    'Gaussian': epsilon**3*(3-6*a**2+a**4)*exp(-(a**2/2))*sqrt(pi/2),
                    'C0 Matern': epsilon**3*(1+abs(a))*exp(-abs(a)),
                    'C2 Matern': epsilon**3/6*(3 +3*abs(a)-6*abs(a)**2+abs(a)**3)*exp(-abs(a)),
                    'C4 Matern': epsilon**3/30*(45 +45*abs(a)-15*abs(a)**3-5*abs(a)**4+abs(a)**5)*exp(-abs(a)),
                    'C6 Matern': epsilon**3/140*(2835 +2835*abs(a)+630*abs(a)**2-315*abs(a)**3-210*abs(a)**4-42*abs(a)**5+abs(a)**7)*exp(-abs(a)),
                    'Inverse Quadratic': 48*(16 +5*a**2*(-8 + a**2))*pi*epsilon**3/((4 + a**2)**5)
                    }
        
        out_val = rbf_D2_switch.get(expansion_type)
        
    return out_val


def gamma_to_x(gamma_vec, tau_vec, expansion_type, epsilon): ## double check this to see if the function is correct
    
    """
    this function mapps gamma_vec back to x, knowing that x = gamma_vec in the piecewise linear case
    """
    
    if expansion_type == 'PWL':
        x_vec = gamma_vec
        
    else:        
        rbf = rbf_switch.get(expansion_type)
        
        N_taus = tau_vec.size
        B = np.zeros([N_taus, N_taus])
        
        for p in range(0, N_taus):
            for q in range(0, N_taus):
                delta_log_tau = log(tau_vec[p])-log(tau_vec[q])
                B[p,q] = rbf(delta_log_tau, epsilon)
                
        B = 0.5*(B+B.T)
                
        x_vec = np.linalg.solve(B, gamma_vec)
            
    return x_vec


def x_to_gamma(x_vec, tau_map_vec, tau_vec, expansion_type, epsilon): # double check this to see if the function is correct
    
    """
    this function mapps x back to gamma_vec, knowing that gamma_vec = 0 in the piecewise linear case
    """
    
    if expansion_type == 'PWL':
        gamma_vec = x_vec
        out_tau_vec = tau_vec

    else:
        rbf = rbf_switch.get(expansion_type)
        
        N_taus = tau_vec.size 
        N_tau_map = tau_map_vec.size
        gamma_vec = np.zeros([N_tau_map, 1])

        B = np.zeros([N_tau_map, N_taus])
        
        for p in range(0, N_tau_map):
            for q in range(0, N_taus):
                delta_log_tau = log(tau_map_vec[p])-log(tau_vec[q])
                B[p,q] = rbf(delta_log_tau, epsilon)              

        gamma_vec = B@x_vec
        out_tau_vec = tau_map_vec 
        
    return gamma_vec


def compute_A_re(freq_vec, tau_vec, expansion_type, epsilon, brute_force):
    
    """
    this function computes the matrix A_re
    """ 
    
    # number of frequencies and timescales
    N_freqs = freq_vec.size 
    N_taus = tau_vec.size 

    # angular frequencies
    omega_vec = 2.*pi*freq_vec

    # define the A_re output matrix
    out_A_re = np.zeros((N_freqs, N_taus)) # size N_freqs*N_taus
    
    # frequency stats
    std_diff_freq = np.std(np.diff(np.log(1/freq_vec)))
    mean_diff_freq = np.mean(np.diff(np.log(1/freq_vec)))

    toeplitz_trick = std_diff_freq/mean_diff_freq<0.01 and N_freqs == N_taus 

    if toeplitz_trick and expansion_type != 'PWL' and brute_force == False: # use Toeplitz trick
        R = np.zeros(N_taus)
        C = np.zeros(N_freqs)
        
        for p in range(0, N_freqs):
            C[p] = g_re(freq_vec[p], tau_vec[0], expansion_type, epsilon)
        
        for q in range(0, N_taus):
            R[q] = g_re(freq_vec[0], tau_vec[q], expansion_type, epsilon)        
                        
        out_A_re= toeplitz(C,R) 

    else: # use brute force
        for p in range(0, N_freqs):
            for q in range(0, N_taus):
            
                if expansion_type == 'PWL':                
                    if q == 0:
                        out_A_re[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                    elif q == N_taus-1:
                        out_A_re[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                    else:
                        out_A_re[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])                    
                
                else:
                    out_A_re[p, q]= g_re(freq_vec[p], tau_vec[q], expansion_type, epsilon)

    return out_A_re


def compute_A_im(freq_vec, tau_vec, expansion_type, epsilon, brute_force):
    
    """
    this function computes the matrix A_im
    """

    # number of frequencies and timescales
    N_freqs = freq_vec.size
    N_taus = tau_vec.size

    # angular frequencies
    omega_vec = 2.*pi*freq_vec

    # define the A_im output matrix
    out_A_im = np.zeros((N_freqs, N_taus)) # size N_freqs*N_taus
    
    # frequency stats
    std_diff_freq = np.std(np.diff(np.log(1/freq_vec)))
    mean_diff_freq = np.mean(np.diff(np.log(1/freq_vec)))

    toeplitz_trick = std_diff_freq/mean_diff_freq<0.01 and N_freqs == N_taus 
       
    if toeplitz_trick and expansion_type != 'PWL' and brute_force == False: # use Toeplitz trick
        R = np.zeros(N_taus)
        C = np.zeros(N_freqs)
        
        for p in range(0, N_freqs):
            C[p] = - g_im(freq_vec[p], tau_vec[0], expansion_type, epsilon)
        
        for q in range(0, N_taus):  
            R[q] = - g_im(freq_vec[0], tau_vec[q], expansion_type, epsilon)        
                        
        out_A_im = toeplitz(C,R) 

    else: # use brute force
        for p in range(0, N_freqs):
            for q in range(0, N_taus):
            
                if expansion_type == 'PWL':                
                    if q == 0:
                        out_A_im[p, q] = - 0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                    elif q == N_taus-1:
                        out_A_im[p, q] = - 0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                    else:
                        out_A_im[p, q] = - 0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])                    
                
                else:
                    out_A_im[p, q]= - g_im(freq_vec[p], tau_vec[q], expansion_type, epsilon)
 
    return out_A_im


def compute_M_D1(tau_vec, expansion_type, epsilon, brute_force):
    
    """
    this function computes the matrix M, which contains the inner products of the first derivatives of the RBF functions used in the      
    expansion; the size of the matrix M is NxN with N the size of the vector tau
    """
    
    N_taus = tau_vec.size # number of collocation points, i.e., the number of timescales

    out_M = np.zeros([N_taus, N_taus]) # the size of the matrix M is N_tausxN_taus with N_taus the number of collocation points
    
    # statistics about the log timescales
    std_diff_tau = np.std(np.diff(np.log(tau_vec)));
    mean_diff_tau = np.mean(np.diff(np.log(tau_vec)));

    toeplitz_trick = std_diff_tau/mean_diff_tau<0.01

    if toeplitz_trick and expansion_type != 'PWL' and brute_force == False: # we use the Toeplitz trick to compute the matrix M
        R = np.zeros(N_taus)
        C = np.zeros(N_taus)
        
        for n in range(0,N_taus):
            C[n] = inner_prod_rbf_D1(tau_vec[n], tau_vec[0], expansion_type, epsilon) 
            
        for m in range(0,N_taus):
            R[m] = inner_prod_rbf_D1(tau_vec[0], tau_vec[m], expansion_type, epsilon)    
        
        out_M = toeplitz(C,R) 
         
    elif expansion_type == 'PWL': # piecewise linear discretization
        out_L_temp = np.zeros([N_taus-1, N_taus])
        
        for iter_freq_n in range(0, N_taus-1):
            delta_loc = log(tau_vec[iter_freq_n+1]/tau_vec[iter_freq_n])
            out_L_temp[iter_freq_n,iter_freq_n] = -1/delta_loc
            out_L_temp[iter_freq_n,iter_freq_n+1] = 1/delta_loc

        out_M = out_L_temp.T@out_L_temp
    
    else: # we compute the rbf with brute force
        print('brute force')
        for n in range(0, N_taus):
            for m in range(0, N_taus):            
                out_M[n,m] = inner_prod_rbf_D1(tau_vec[n], tau_vec[m], expansion_type, epsilon)
               
    return out_M


def compute_M_D2(tau_vec, expansion_type, epsilon, brute_force):
    
    """
    this function computes the matrix M, which contains the inner products of the second derivatives of the RBF functions used in the      
    expansion
    """

    N_taus = tau_vec.size # number of collocation points, i.e., the number of timescales
    
    out_M = np.zeros([N_taus, N_taus]) # the size of the matrix M is N_tausxN_taus with N_taus the number of collocation points
    
    # statistics about the log timescales
    std_diff_tau = np.std(np.diff(np.log(tau_vec))); 
    mean_diff_tau = np.mean(np.diff(np.log(tau_vec))); 
        
    toeplitz_trick = std_diff_tau/mean_diff_tau<0.01

    if toeplitz_trick and expansion_type != 'PWL' and brute_force == False: # we use the Toeplitz trick to compute the matrix M
        R = np.zeros(N_taus)
        C = np.zeros(N_taus)
        
        for n in range(0,N_taus):
            C[n] = inner_prod_rbf_D2(tau_vec[n], tau_vec[0], expansion_type, epsilon)
            
        for m in range(0,N_taus):
            R[m] = inner_prod_rbf_D2(tau_vec[0], tau_vec[m], expansion_type, epsilon)
        
        out_M = toeplitz(C,R) 
         
    elif expansion_type == 'PWL': # piecewise linear discretization
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
    
    else: # we compute the rbf with brute force
        print('brute force')
        for n in range(0, N_taus):
            for m in range(0, N_taus):            
                out_M[n,m] = inner_prod_rbf_D2(tau_vec[n], tau_vec[m], expansion_type, epsilon)
    
    return out_M


def assemble_A(freq_vec, tau_vec, expansion_type, epsilon, include_RL, brute_force=False):
    
    """
    this function assembles the matrices A_re, A_im, and A with the possibility to include a resistance, an inductance, or both
    """

    N_f = freq_vec.size
    A_re = compute_A_re(freq_vec, tau_vec, expansion_type, epsilon, brute_force)
    A_im = compute_A_im(freq_vec, tau_vec, expansion_type, epsilon, brute_force)

    A_re_R_inf = np.ones((N_f, 1))
    A_im_R_inf = np.zeros((N_f, 1))

    A_re_L_0 = np.zeros((N_f, 1))
    A_im_L_0 = 2*pi*freq_vec.reshape((N_f, 1))

    if include_RL == 'R':
        A_re = np.hstack((A_re_R_inf, A_re)) 
        A_im = np.hstack((A_im_R_inf, A_im)) 

    elif include_RL == 'L':
        A_re = np.hstack(( A_re_L_0, A_re)) 
        A_im = np.hstack(( A_im_L_0, A_im)) 

    elif include_RL == 'R+L' or 'L+R':
        A_re = np.hstack(( A_re_R_inf, A_re_L_0, A_re))
        A_im = np.hstack(( A_im_R_inf, A_im_L_0, A_im))

    A = np.vstack((A_re, A_im))

    return A_re, A_im, A


def assemble_M(tau_vec, expansion_type, epsilon, derivative, include_RL, brute_force = False):
    
    """
    this function assembles the matrice M according to the derivative order and with the possibility 
    to include a resistance, an inductance, or both
    """

    N_tau = tau_vec.shape[0]

    if derivative == '1st':
        M_temp = compute_M_D1(tau_vec, expansion_type, epsilon, brute_force)
    elif derivative == '2nd':
        M_temp = compute_M_D2(tau_vec, expansion_type, epsilon, brute_force)
    else:
        M_temp = compute_M_D1(tau_vec, expansion_type, epsilon, brute_force)      

    if include_RL == 'R': # with a resistance 
        M = np.zeros((N_tau+1, N_tau+1))
        M[1:, 1:] = M_temp
        
    elif include_RL == 'L': # with an inductance
        M = np.zeros((N_tau+1, N_tau+1))
        M[1:, 1:] = M_temp

    elif include_RL == 'R+L' or 'L+R': # with both an inductance and a resistance
        M = np.zeros((N_tau+2, N_tau+2)) 
        M[2:, 2:] = M_temp

    else: # without any specification
        M = M_temp

    return M


def compute_L1(tau_vec):
    
    N_taus = tau_vec.size
    out_L1 = np.zeros((N_taus-1, N_taus+2))
    
    out_L_temp = np.zeros((N_taus-1, N_taus))
    
    for p in range(0,N_taus-1):
            delta_loc = log(tau_vec[p+1]/tau_vec[p])
            out_L_temp[p,p] = -1/delta_loc
            out_L_temp[p,p+1] = 1/delta_loc

    out_L1[:,2:] = out_L_temp 
            
    return out_L1


def compute_L2(tau_vec):
    
    N_taus = tau_vec.size
    out_L2 = np.zeros((N_taus-2, N_taus+2))
    
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
    
    out_L2[:,2:] = out_L_temp 
        
    return out_L2


def assemble_Z(Z_exp):
    
    """
    this function stacks the real and impaginary parts of the impedance in a vector Z
    """
    
    Z_re = Z_exp.real
    Z_im = Z_exp.imag

    Z = np.hstack((Z_re, Z_im))
    
    return Z_re, Z_im, Z


def is_PD(A):
    
    """
    this function checks if a matrix A is positive-definite using Cholesky transform
    """

    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

    
def nearest_PD(A):
    
    """
    this function finds the nearest positive definite matrix of a matrix A; this code is based on John D'Errico's "nearestSPD" 
    code on Matlab [1], which leverages [2].
    References:
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    
    B = (A + A.T)/2
    _, Sigma_mat, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(Sigma_mat), V))

    A_nPD = (B + H) / 2
    A_symm = (A_nPD + A_nPD.T) / 2

    k = 1
    I = np.eye(A_symm.shape[0])

    while not is_PD(A_symm): # the Matlab function chol accepts matrices with eigenvalue = 0, but numpy does not so we replace
        # the matlab function eps(min_eig) with the following one
        eps = np.spacing(np.linalg.norm(A_symm))
        min_eig = min(0, np.min(np.real(np.linalg.eigvals(A_symm))))
        A_symm += I * (-min_eig * k**2 + eps)
        k += 1

    return A_symm


def compute_GCV(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
    this function computes the minimizer of the generalized cross-validation approach    
    """
    
    lambda_value = exp(log_lambda)
    
    A = np.concatenate((A_re, A_im), axis = 0) # concatenate the matrices A_re and A_im in the matrix A
    Z = np.concatenate((Z_re, Z_im), axis = 0) # concatenate the vectors Z_re and Z_im in the vector Z
    
    n_cv = Z.shape[0] # n_cv = 2*N_freqs with N_freqs the number of EIS probed frequencies
    A_agm = A.T@A + lambda_value*M # A_agm = A^T*A + lambda*M
    
    if (is_PD(A_agm)==False):
        A_agm = nearest_PD(A_agm) 
        
    L_agm = cholesky(A_agm) # Cholesky transform to inverse A_agm
    inv_L_agm = np.linalg.inv(L_agm)
    inv_A_agm = inv_L_agm.T@inv_L_agm # inverse of A_agm
    K = A@inv_A_agm@A.T  
    
    # Score (equation (13) of the main manuscript)
    GCV_num = 1/n_cv*norm((np.eye(n_cv) - K)@Z)**2 # numerator
    GCV_dom = (1/n_cv*np.trace(np.eye(n_cv) - K))**2 # denominator
    
    GCV_score = GCV_num/GCV_dom
    
    return GCV_score


def compute_LC(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
    this function computes the minimizer for the L-curve approach
    """
    
    lambda_value = exp(log_lambda)
    
    A = np.concatenate((A_re, A_im), axis = 0) # concatenate the matrices A_re and A_im in the matrix A
    Z = np.concatenate((Z_re, Z_im), axis = 0) # concatenate the vectors Z_re and Z_im in the vector Z
    
    # numerator eta_num of the first derivative of eta
    A_agm = A.T@A + lambda_value*M # same matrix as in compute_GCV (line 644)
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
    
    # numerator theta_num of the first derivative of theta
    theta_num  = eta_num
    
    # denominator theta_denom of the first derivative of theta
    A_LC_d = A@(inv_A_agm.T@inv_A_agm)@A.T
    theta_denom = Z.T@A_LC_d@Z
    
    # derivative of theta 
    theta_prime = -(theta_num)/theta_denom
    
    # numerator LC_num of the minimizer LC for the L curve
    a_sq = (eta_num/(eta_denom*theta_denom))**2
    p = (Z.T@(inv_A_agm_d.T@inv_A_agm_d)@Z)*theta_denom
    m = (2*lambda_value*Z.T@((inv_A_agm_d.T@inv_A_agm_d)@inv_A_agm_d)@Z)*theta_denom
    q = (2*lambda_value*Z.T@(inv_A_agm_d.T@inv_A_agm_d)@Z)*eta_num 
    LC_num = a_sq*(p+m-q)

    # denominator LC_denom of the minimizer LC for the L curve
    LC_denom = ((eta_prime)**2 + (theta_prime)**2)**(3/2)
    
    # Score (equation (19) of the main manuscript)
    LC_score = LC_num/LC_denom 
    
    return -LC_score # minus since we wish to maximize LC_score


def compute_re_im_cv(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
    this function computes the minimizer for the real-imaginary cross-validation discrepancy
    """
    
    lambda_value = exp(log_lambda)
    
    # non-negativity constraint on the DRT gmma
    lb = np.zeros([Z_re.shape[0]+1])
    bound_mat = np.eye(lb.shape[0]) 
    
    # quadratic programming through cvxopt
    H_re, c_re = quad_format(A_re, Z_re, M, lambda_value)
    gamma_ridge_re = cvxopt_solve_qpr(H_re, c_re, -bound_mat, lb)
    H_im, c_im = quad_format(A_im, Z_im, M, lambda_value)
    gamma_ridge_im = cvxopt_solve_qpr(H_im, c_im, -bound_mat, lb)
    
    # stacking the resistance R and inductance L on top of gamma_ridge_im and gamma_ridge_re, repectively
    gamma_ridge_re_cv = np.concatenate((np.array([0, gamma_ridge_re[1]]), gamma_ridge_im[2:]))
    gamma_ridge_im_cv = np.concatenate((np.array([gamma_ridge_im[0], 0]), gamma_ridge_re[2:]))
    
    # Score (equation (17) of the main manuscript)
    re_im_cv_score = norm(Z_re - A_re@gamma_ridge_re_cv)**2+norm(Z_im-A_im@gamma_ridge_im_cv)**2
    
    return re_im_cv_score


def compute_kf_cv(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
    this function computes the minimizer for the k-fold cross-validation approach
    """

    lambda_value = exp(log_lambda)
    
    # non-negativity constraint on the DRT gamma
    lb = np.zeros([Z_re.shape[0]+1])
    bound_mat = np.eye(lb.shape[0])
    
    # parameters for kf
    N_splits = 5 # for N_splits equal to N_freq, it is known as leave-one-out cross-validation
    random_state = 34054 + compute_kf_cv.counter*100  # change the random state for each experiment
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
        
        # step 3: update of the kf minimizer    
        kf_cv += 1/Z_re_test.shape[0]*(norm(Z_re_test-A_re_test@gamma_ridge)**2 + norm(Z_im_test-A_im_test@gamma_ridge)**2)
    
    # Score (Section S1.2 of the supplementary information)
    kf_cv_score = kf_cv/N_splits
    return kf_cv_score
compute_kf_cv.counter = 0


def cvxopt_solve_qpr(P, q, G=None, h=None, A=None, b=None):
    
    """
    this function formats a numpy matrix to cvxopt matrix, conducts the
    quadratic programming with cvxopt, and outputs the optimum as a numpy array
    """
    
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
         args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
        
    cvxopt.solvers.options['abstol'] = 1e-15
    cvxopt.solvers.options['reltol'] = 1e-15
    try:
        sol = cvxopt.solvers.qp(*args)
    except:
        sol = cvxopt.solvers.qp(*args, kktsolver='ldl', options={'kktreg':1e-15})
    
    if 'optimal' not in sol['status']:
        return None
    
    return np.array(sol['x']).reshape((P.shape[1],))


def compute_mGCV(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
    this function computes the minimizer of the modified generalized cross-validation approach
    """
    
    lambda_value = exp(log_lambda)
    
    A = np.concatenate((A_re,A_im), axis = 0) # concatenate the matrices A_re and A_im in the matrix A
    Z = np.concatenate((Z_re,Z_im), axis = 0) # concatenate the vectors Z_re and Z_im in the vector Z
    
    n_cv = Z.shape[0] # n_cv = 2*N_freqs with N_freqs the number of EIS proved frequencies
    A_agm = A.T@A + lambda_value*M # same matrix as in compute_GCV (line 644)
    
    if (is_PD(A_agm)==False):
        A_agm = nearest_PD(A_agm)
        
    L_agm = cholesky(A_agm) # Cholesky transform to inverse A_agm
    inv_L_agm = np.linalg.inv(L_agm)
    inv_A_agm = inv_L_agm.T@inv_L_agm # inverse of A_agm
    K = A@inv_A_agm@A.T
    
    # computation of rho with compute_rho
    """
    M.A. Lukas, F.R. de Hoog, R.S. Anderssen, Practical use of robust GCV and modified GCV for spline smoothing, Comput. Stat. 31 (2016)       269–289.
    
    rho = 1.3 if M(number of frequencies < 50) && rho = 2 (if M> 50)
    """
    rho = 2 # (equation (15) of the main manuscript)
    
    # Score (equation (14) of the main manucript)
    mGCV_num = 1/n_cv*norm((np.eye(n_cv) - K)@Z)**2 # numerator
    mGCV_dom = ((1/n_cv)*(np.trace(np.eye(n_cv) - rho*K)))**2 # denominator
    mGCV_score = mGCV_num/mGCV_dom
    
    return mGCV_score


def compute_rGCV(log_lambda, A_re, A_im, Z_re, Z_im, M):
    
    """
    this function computes the minimizer of the robust generalized cross-validation approach
    """
     
    lambda_value = exp(log_lambda)
    
    A = np.concatenate((A_re, A_im), axis = 0) # concatenate the matrices A_re and A_im in the matrix A
    Z = np.concatenate((Z_re, Z_im), axis = 0) # concatenate the vectors Z_re and Z_im in the vector Z
    
    n_cv = Z.shape[0] # n_cv = 2*N_freqs with N_freqs the number of EIS proved frequencies
    A_agm = A.T@A + lambda_value*M # same matrix as in compute_GCV (line 644)

    if (is_PD(A_agm)==False):
        A_agm = nearest_PD(A_agm)

    L_agm = cholesky(A_agm) # Cholesky transform to inverse A_agm
    inv_L_agm = np.linalg.inv(L_agm)
    inv_A_agm = inv_L_agm.T@inv_L_agm # inverse of A_agm
    K = A@inv_A_agm@A.T
    
    # Score (equation (16) of the main manuscript)
    rGCV_num = 1/n_cv*norm((np.eye(n_cv) - K)@Z)**2
    rGCV_dom = ((1/n_cv)*(np.trace(np.eye(n_cv) - K)))**2
    rGCV = rGCV_num/rGCV_dom
    
    # Based on rho value selected above, we compute the robust parameter, xi, according to the relation from Lukas et al.
    """
    M.A. Lukas, F.R. de Hoog, R.S. Anderssen, Practical use of robust GCV and modified GCV for spline smoothing, Comput. Stat. 31 (2016)       269–289
    
    """
    xi = 0.3
    # parameter mu_2
    mu_2 = (1/n_cv)*np.trace(K.T@K)
    
    # minimizer
    rGCV_score = (xi + (1-xi)*mu_2)*rGCV
        
    return rGCV_score


def cvxpy_solve_qp(Z, A, M, lambda_val):   
    
    """
    this function minimizes the problem (Z-Ax)^T.(Z-Ax) + lambda*(x^T.M.x) with lambda given as imput 
    and under the onstraint that x is positive
    """

    N_out = A.shape[1]
    x = cp.Variable(shape = N_out) 
    

    P = A.T@A+lambda_val*M
    # P = (P.T+P)/2
    c = -2*Z.T@A
    
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + c@x), [x >= 0])
    prob.solve(verbose = True, eps_abs = 1E-10, eps_rel = 1E-10, sigma = 1.00e-08, 
               max_iter = 200000, eps_prim_inf = 1E-5, eps_dual_inf = 1E-5)

    gamma = x.value
    
    return gamma


def quad_format(A, Z, M, lambda_value):
    
    """
    this function reformats the DRT regression as a quadratic program using either the real or imaginary part of the impedance
    """
    
    H = 2*(A.T@A+lambda_value*M)
    H = (H.T+H)/2
    c = -2*Z.T@A
    
    return H,c


def quad_format_combined(A_re, A_im, Z_re, Z_im, M, lambda_value): 
    
    """
    this function reformats the DRT regression as a quadratic program using both parts of the impedance
    """
    
    H = 2*((A_re.T@A_re+A_im.T@A_im)+lambda_value*M)
    H = (H.T+H)/2
    c = -2*(Z_im.T@A_im+Z_re.T@A_re)

    return H,c


def optimal_lambda(A_re, A_im, Z_re, Z_im, M, log_lambda_0, cv_type):
    
    """
    this function returns the regularization coefficient given as inputs an initial guess and a regularization method 
    """
    # bound for the lambda values
    bnds = [(log(10**-7),log(10**0))] 
    
    # GCV cross-validation
    if cv_type == 'GCV': 
        res = minimize(compute_GCV, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('GCV')
    
    # mGCV cross-validation
    elif cv_type == 'mGCV': 
        res = minimize(compute_mGCV, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('mGCV')
        
    # rGCV cross-validation
    elif cv_type == 'rGCV': 
        res = minimize(compute_rGCV, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('rGCV')  
    
    # L-curve method
    elif cv_type == 'LC':
        res = minimize(compute_LC, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('LC')
    
    # re-im cross-validation
    elif cv_type == 're-im':
        res = minimize(compute_re_im_cv, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('re-im')
        
    # k-fold cross-validation
    else: #elif cv_type == 'kf':  
        res = minimize(compute_kf_cv, log_lambda_0, args=(A_re, A_im, Z_re, Z_im, M), options={'disp': True, 'maxiter': 2000}, bounds = bnds, method = 'SLSQP')
        print('kf') 
    
    lambda_value = exp(res.x)
    
    return lambda_value


def pretty_plot(width=8, height=None, plt=None, dpi=None, color_cycle=("qualitative", "Set1_9")):
    """
    This code is bollowed from pymatgen to produce high quality figures. This needs further polishing later
    Args:
        width (float): Width of plot in inches. Defaults to 8in.
        height (float): Height of plot in inches. Defaults to width * golden
            ratio.
        plt (matplotlib.pyplot): If plt is supplied, changes will be made to an
            existing plot. Otherwise, a new plot will be created.
        dpi (int): Sets dot per inch for figure. Defaults to 300.
        color_cycle (tuple): Set the color cycle for new plots to one of the
            color sets in palettable. Defaults to a qualitative Set1_9.
    Returns:
        Matplotlib plot object with properly sized fonts.
    """
    ticksize = int(width * 2.5)

    golden_ratio = (sqrt(5) - 1) / 2

    if not height:
        height = int(width * golden_ratio)

    if plt is None:
        import matplotlib.pyplot as plt
        import importlib
        mod = importlib.import_module("palettable.colorbrewer.%s" %
                                      color_cycle[0])
        colors = getattr(mod, color_cycle[1]).mpl_colors
#        from cycler import cycler

        plt.figure(figsize=(width, height), facecolor="w", dpi=dpi)
        ax = plt.gca()
#        ax.set_prop_cycle(cycler('color', colors))
    else:
        
        fig = plt.gcf()
        fig.set_size_inches(width, height)
        
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    ax = plt.gca()
    ax.set_title(ax.get_title(), size=width * 4)

    labelsize = int(width * 3)

    ax.set_xlabel(ax.get_xlabel(), size=labelsize)
    ax.set_ylabel(ax.get_ylabel(), size=labelsize)

    return plt
