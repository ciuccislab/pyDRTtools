# -*- coding: utf-8 -*-
__authors__ = 'Francesco Ciucci, Ting Hei Wan, Adeleke Maradesa, Baptiste Py'
__date__ = '13th November 2023'

"""
    Several Python packages are required, namely numpy, pandas, math, scipy, and matplotlib.
"""


# Maths and data related packages
import numpy as np
from numpy import log, log10, sqrt
import pandas as pd
from math import pi
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# pyDRTtools related package
from . import basics 
import importlib
importlib.reload(basics)
from . import BHT 
from . import HMC 

class EIS_object(object):
    
    # The EIS_object class stores the input data and the DRT result.
      
    def __init__(self, freq, Z_prime, Z_double_prime):
        
        """
        This function define an EIS_object.
        Inputs:
            freq: frequency of the EIS measurement
            Z_prime: real part of the impedance
            Z_double_prime: imaginery part of the impedance
        """
        
        # define an EIS_object
        self.freq = freq
        self.Z_prime = Z_prime
        self.Z_double_prime = Z_double_prime
        self.Z_exp = Z_prime + 1j*Z_double_prime
        
        # keep a copy of the original data
        self.freq_0 = freq
        self.Z_prime_0 = Z_prime
        self.Z_double_prime_0 = Z_double_prime
        self.Z_exp_0 = Z_prime + 1j*Z_double_prime
        
        self.tau = 1/freq # we assume that the collocation points equal to 1/freq as default
        self.tau_fine  = np.logspace(log10(self.tau.min())-0.5,log10(self.tau.max())+0.5,10*freq.shape[0])            

        self.method = 'none'
    
    @classmethod
    def from_file(cls,filename):
        
        if filename.endswith('.csv'): # import from csv file
            data = pd.read_csv(filename, header=None).to_numpy()
            freq = data[:, 0]
            Z_prime = data[:, 1]
            Z_double_prime = data[:, 2]
        
        elif filename.endswith('.txt'): # import from txt file
            data = np.loadtxt(filename)
            freq = data[:, 0]
            Z_prime = data[:, 1]
            Z_double_prime = data[:, 2]
    
        return cls(freq, Z_prime, Z_double_prime)
    
    def plot_DRT(self): # plot the DRT result
        
        basics.pretty_plot(4,4)
        plt.rc('font', family='serif', size=15)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.rc('text', usetex=True)    
        
        if self.method == 'simple':    
            plt.plot(self.out_tau_vec, self.gamma, 'k')
            y_min = 0
            y_max = max(self.gamma)
            
        elif self.method == 'credit':
            plt.fill_between(self.out_tau_vec, self.lower_bound, self.upper_bound,  facecolor='lightgrey')
            plt.plot(self.out_tau_vec, self.gamma, color='black', label='MAP')
            plt.plot(self.out_tau_vec, self.mean, color='blue', label='mean')
            plt.plot(self.out_tau_vec, self.lower_bound, color='black', linewidth=1)
            plt.plot(self.out_tau_vec, self.upper_bound, color='black', linewidth=1)
            plt.legend(frameon=False, fontsize = 15)
            y_min = 0
            y_max = max(self.upper_bound)
            
        elif self.method == 'BHT':    
            plt.semilogx(self.out_tau_vec, self.mu_gamma_fine_re, 'b', linewidth=1)
            plt.semilogx(self.out_tau_vec, self.mu_gamma_fine_im, 'k', linewidth=1)
            y_min = min(np.concatenate((self.mu_gamma_fine_re, self.mu_gamma_fine_im)))
            y_max = max(np.concatenate((self.mu_gamma_fine_re, self.mu_gamma_fine_im)))
        
        else:
            return
        
        plt.xscale('log')
        plt.xlim(self.out_tau_vec.min(), self.out_tau_vec.max())
        plt.ylim(y_min, y_max*1.1)
        plt.xlabel(r'$f/{\rm Hz}$', fontsize=20)
        plt.ylabel(r'$\gamma(\tau)/\Omega$', fontsize=20)
    
        plt.show()


def simple_run(entry, rbf_type = 'Gaussian', data_used = 'Combined Re-Im Data', induct_used = 1, der_used = '1st order', cv_type = 'GCV', shape_control = 'FWHM Coefficient', coeff = 0.5):
    
    """
    This function enables to compute the DRT using ridge regression (also known as Tikhonov regression)
    References:
        T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, Influence of the discretization methods on the distribution of relaxation times deconvolution: Implementing radial basis functions with DRTtools, Electrochimica Acta 184 (2015) 483-499.
    Inputs:
        entry: an EIS spectrum
        rbf_type: discretization function
        data_used: part of the EIS spectrum used for regularization
        induct_used: treatment of the inductance part
        der_used: order of the derivative considered for the M matrix
        cv_type: regularization method used to select the regularization parameter for ridge regression 
        shape_control: option for controlling the shape of the radial basis function (RBF) 
        coeff: magnitude of the shape control
    """
    
    # Step 1: define the matrices
    
    # Step 1.1: define the optimization bounds
    N_freqs = entry.freq.shape[0]
    N_taus = entry.tau.shape[0]
    ###
    entry.b_re = entry.Z_exp.real
    entry.b_im = entry.Z_exp.imag
    # Step 1.2: compute epsilon
    entry.epsilon = basics.compute_epsilon(entry.freq, coeff, rbf_type, shape_control)
    
    # Step 1.3: compute A matrix
    entry.A_re_temp = basics.assemble_A_re(entry.freq, entry.tau, entry.epsilon, rbf_type, flag1='simple', flag2='impedance')
    entry.A_im_temp = basics.assemble_A_im(entry.freq, entry.tau, entry.epsilon, rbf_type, flag1='simple', flag2='impedance')
    
    # Step 1.4: compute M matrix
    if der_used == '1st order':
        entry.M_temp = basics.assemble_M_1(entry.tau, entry.epsilon, rbf_type, flag='simple')
    elif der_used == '2nd order':
        entry.M_temp = basics.assemble_M_2(entry.tau, entry.epsilon, rbf_type, flag='simple')
    
    # Step 2: conduct ridge regularization
    if data_used == 'Combined Re-Im Data': # select both parts of the impedance for the simple run
 
        if induct_used == 0 or induct_used == 2: # without considering the inductance
            
            N_RL = 1 # N_RL length of resistance plus inductance
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:,N_RL:] = entry.A_re_temp
            entry.A_re[:,0] = 1
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:,N_RL:] = entry.A_im_temp
            
            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
            
        elif induct_used == 1: #considering the inductance
            N_RL = 2
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            entry.A_re[:,1] = 1
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            entry.A_im[:,0] = 2*pi*entry.freq

            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
        
        # optimally select the regularization level
        log_lambda_0 = log(10**-3) # initial guess for lambda
        lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, log_lambda_0, cv_type) 

        # recover the DRT using cvxopt
        H_combined,c_combined = basics.quad_format_combined(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, lambda_value)
        ## Enforce positivity constraint
        lb = np.zeros([entry.b_re.shape[0]+1]) 
        bound_mat = np.eye(lb.shape[0])
        ###
        x = basics.cvxopt_solve_qpr(H_combined, c_combined,-bound_mat,lb) # using cvxopt
        # prepare for HMC sampler, it will be used if needed
        entry.mu_Z_re = entry.A_re@x
        entry.mu_Z_im = entry.A_im@x
        entry.res_re = entry.mu_Z_re-entry.b_re
        entry.res_im = entry.mu_Z_im-entry.b_im

        # only consider std of residuals in both parts
        sigma_re_im = np.std(np.concatenate([entry.res_re,entry.res_im]))
        inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
    
        Sigma_inv = (entry.A_re.T@inv_V@entry.A_re) + (entry.A_im.T@inv_V@entry.A_im) + (lambda_value/sigma_re_im**2)*entry.M
        mu_numerator = entry.A_re.T@inv_V@entry.b_re + entry.A_im.T@inv_V@entry.b_im
        
    elif data_used == 'Im Data': # select the imaginary part of the impedance for the simple run
        
        if induct_used == 0 or induct_used == 2: # without considering the inductance
            N_RL = 0 # N_RL length of resistance plus inductance
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            
            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
            
        elif induct_used == 1: # considering the inductance
            N_RL = 1
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            entry.A_im[:,0] = 2*pi*entry.freq
            
            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
        
        # optimally select the regularization level
        log_lambda_0 = log(10**-3) # initial guess for lambda
        lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, log_lambda_0, cv_type) 
        
        ##
        #### Positivity Constraints
        ##
        lb = np.zeros([entry.b_re.shape[0]+1]) 
        bound_mat = np.eye(lb.shape[0])
        
        # recover the DRT using cvxopt
        H_im, c_im = basics.quad_format_separate(entry.A_im, entry.b_im, entry.M, lambda_value)

        x = basics.cvxopt_solve_qpr(H_im, c_im,-bound_mat,lb) # using cvxopt
        # prepare for HMC sampler
        entry.mu_Z_re = entry.A_re@x
        entry.mu_Z_im = entry.A_im@x
        entry.res_re = entry.mu_Z_re-entry.b_re
        entry.res_im = entry.mu_Z_im-entry.b_im
        
        # only consider std of residuals in the imaginary part
        sigma_re_im = np.std(entry.res_im)
        inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
        
        Sigma_inv = (entry.A_im.T@inv_V@entry.A_im) + (lambda_value/sigma_re_im**2)*entry.M
        mu_numerator = entry.A_im.T@inv_V@entry.b_im
        
    elif data_used == 'Re Data': # select the real part of the impedance for the simple run
        
        N_RL = 1
        entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
        entry.A_re[:, N_RL:] = entry.A_re_temp
        entry.A_re[:,0] = 1
        
        entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
        entry.A_im[:, N_RL:] = entry.A_im_temp

        entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
        entry.M[N_RL:,N_RL:] = entry.M_temp
        
        # optimally select the regularization level
        log_lambda_0 = log(10**-3) # initial guess for lambda
        lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, log_lambda_0, cv_type) 
        
        # recover the DRT using cvxopt 
        H_re,c_re = basics.quad_format_separate(entry.A_re, entry.b_re, entry.M, lambda_value)
        ## Enforce negativity constraint
        lb = np.zeros([entry.b_re.shape[0]+1]) 
        bound_mat = np.eye(lb.shape[0])
        # recovered DRT
        x = basics.cvxopt_solve_qpr(H_re, c_re, -bound_mat,lb) # using cvxopt
        # prepare for HMC sampler
        entry.mu_Z_re = entry.A_re@x
        entry.mu_Z_im = entry.A_im@x       
        entry.res_re = entry.mu_Z_re-entry.b_re
        entry.res_im = entry.mu_Z_im-entry.b_im
        
        # only consider std of residuals in the real part
        sigma_re_im = np.std(entry.res_re)
        inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
        
        Sigma_inv = (entry.A_re.T@inv_V@entry.A_re) + (lambda_value/sigma_re_im**2)*entry.M
        mu_numerator = entry.A_re.T@inv_V@entry.b_re

    entry.Sigma_inv = (Sigma_inv+Sigma_inv.T)/2
    
    # test if the covariance matrix is positive definite
    if (basics.is_PD(entry.Sigma_inv)==False):
        entry.Sigma_inv = basics.nearest_PD(entry.Sigma_inv) # if not, use the nearest positive definite matrix
    
    L_Sigma_inv = np.linalg.cholesky(entry.Sigma_inv)
    entry.mu = np.linalg.solve(L_Sigma_inv, mu_numerator)
    entry.mu = np.linalg.solve(L_Sigma_inv.T, entry.mu)
    # entry.mu = np.linalg.solve(entry.Sigma_inv, mu_numerator)
    
    # Step 3: obtaining the result of inductance, resistance, and gamma  
    if N_RL == 0: 
        entry.L, entry.R = 0, 0        
    elif N_RL == 1 and data_used == 'Im Data':
        entry.L, entry.R = x[0], 0    
    elif N_RL == 1 and data_used != 'Im Data':
        entry.L, entry.R = 0, x[0]
    elif N_RL == 2:
        entry.L, entry.R = x[0:2]
    
    entry.x = x[N_RL:]
    entry.out_tau_vec, entry.gamma = basics.x_to_gamma(x[N_RL:], entry.tau_fine, entry.tau, entry.epsilon, rbf_type)
    entry.N_RL = N_RL 
    entry.method = 'simple'
    
    return entry


def Bayesian_run(entry, rbf_type = 'Gaussian', data_used = 'Combined Re-Im Data', induct_used = 1, der_used = '1st order', cv_type = 'GCV', shape_control = 'FWHM Coefficient', coeff = 0.5, NMC_sample = 2000):
    
    """
    This function enables to recover the DRT with its uncertainty in a Bayesian framework. 
    References:
        F. Ciucci, C. Chen, Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach, Electrochimica Acta 167 (2015) 439-454.
        M. B. Effat, F. Ciucci, Bayesian and hierarchical Bayesian based regularization for deconvolving the distribution of relaxation times from electrochemical impedance spectroscopy data, Electrochimica Acta 247 (2017) 1117-1129.
    Inputs:
        entry: an EIS spectrum
        rbf_type: discretization function
        data_used: part of the EIS spectrum used for regularization
        induct_used: treatment of the inductance part
        der_used: order of the derivative considered for the M matrix
        cv_type: regularization method used to select the regularization parameter for ridge regression 
        shape_control: option for controlling the shape of the radial basis function (RBF) 
        coeff: magnitude of the shape control
        NMC_sample: number of samples for the HMC sampler
    """
    
    simple_run(entry, rbf_type, data_used, induct_used, 
               der_used, cv_type, shape_control, coeff) 

    # using HMC sampler to sample the truncated Gaussian distribution
    
    # object_A.plot_DRT()
    entry.mu = entry.mu[entry.N_RL:] # reshape to avoid error as
    entry.Sigma_inv = entry.Sigma_inv[entry.N_RL:,entry.N_RL:]
    
    # Step 1: Cholesky Transform instead of direct inverse 
    L_Sigma_inv = np.linalg.cholesky(entry.Sigma_inv)
    L_Sigma_agm = np.linalg.inv(L_Sigma_inv)
    entry.Sigma = L_Sigma_agm.T@L_Sigma_agm
    
    # Step 2: set up the boundary constraints
    F = np.eye(entry.x.shape[0])
    g = np.finfo(float).eps*np.ones(entry.mu.shape[0])
    initial_X = entry.x
    
    # Step 3: use generate_tmg from HMC_exact.py to sample the truncated Gaussian distribution
    entry.Xs = HMC.generate_tmg(F, g, entry.Sigma, entry.mu, initial_X, cov=True, L=NMC_sample)
    entry.lower_bound = np.quantile(entry.Xs[:,501:],.005,axis=1)
    entry.upper_bound = np.quantile(entry.Xs[:,501:],.995,axis=1)
    entry.mean = np.mean(entry.Xs[:,501:],axis=1)    
    
    # Step 4: map array to gamma
    entry.out_tau_vec,entry.lower_bound = basics.x_to_gamma(entry.lower_bound, entry.tau_fine, entry.tau, entry.epsilon, rbf_type)
    entry.out_tau_vec,entry.upper_bound = basics.x_to_gamma(entry.upper_bound, entry.tau_fine, entry.tau, entry.epsilon, rbf_type)
    entry.out_tau_vec,entry.mean = basics.x_to_gamma(entry.mean, entry.tau_fine, entry.tau, entry.epsilon, rbf_type)
    
    entry.method = 'credit'

    return entry


def BHT_run(entry, rbf_type = 'Gaussian', der_used = '1st order', shape_control = 'FWHM Coefficient', coeff = 0.5):
    
    """
    This function enables to assess the compliance of an EIS spectrum to the Kramers-Kronig relations.
    References: 
       J. Liu, T. H. Wan, F. Ciucci, A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores, Electrochimica Acta. 357 (2020) 136864.
       F. Ciucci, The Gaussian process hilbert transform (GP-HT): Testing the consistency of electrochemical impedance spectroscopy data, Journal of the Electrochemical Society. 167-12 (2020) 126503.
    Inputs:
       entry: an EIS spectrum
       rbf_type: discretization function
       der_used: order of the derivative considered for the M matrix
       shape_control: option for controlling the shape of the radial basis function (RBF) 
       coeff: magnitude of the shape control           
    """   
    
    omega_vec = 2*pi*entry.freq
    N_freqs = entry.freq.shape[0]
    N_taus = entry.tau.shape[0]
    
    # Step 1: construct the A matrix
    entry.epsilon = basics.compute_epsilon(entry.freq, coeff, rbf_type, shape_control)
    A_re_temp = basics.assemble_A_re(entry.freq, entry.tau, entry.epsilon, rbf_type, flag1='simple', flag2='impedance')
    A_im_temp = basics.assemble_A_im(entry.freq, entry.tau, entry.epsilon, rbf_type, flag1='simple', flag2='impedance')
    
    # add resistance column and inductance column to A_re and A_im
    entry.A_re = np.append(np.ones([N_freqs,1]), A_re_temp, axis=1)
    entry.A_im = np.append(omega_vec.reshape(N_freqs,1), A_im_temp, axis=1)
    entry.A_H_re = A_re_temp
    entry.A_H_im = A_im_temp  
    entry.b_re = entry.Z_exp.real
    entry.b_im = entry.Z_exp.imag
    
    # Step 2: construct the M matrix
    if der_used == '1st order':
        entry.M_temp = basics.assemble_M_1(entry.tau, entry.epsilon, rbf_type, flag='simple')
    
    elif der_used == '2nd order':
        entry.M_temp = basics.assemble_M_2(entry.tau, entry.epsilon, rbf_type, flag='simple')
    entry.M = np.zeros((N_taus+1, N_taus+1))
    entry.M[1:,1:] = entry.M_temp 
    
    # Step 3: test HT_single_est (try until no error occur for the HT_single_est)
    while True:
        try:
            theta_0 = 10**(6*np.random.rand(3, 1)-3)
            out_dict_real = BHT.HT_single_est(theta_0, entry.Z_exp.real, entry.A_re, entry.A_H_im, entry.M, N_freqs, N_taus)
            theta_0 = out_dict_real['theta']
            out_dict_imag = BHT.HT_single_est(theta_0, entry.Z_exp.imag, entry.A_im, entry.A_H_re, entry.M, N_freqs, N_taus)
            
            break
            
        except:
            print('Error Occur, Try Another Inital Condition')
    
    # Step 4: score the EIS
    entry.out_scores = BHT.EIS_score(theta_0, entry.freq, entry.Z_exp, out_dict_real, out_dict_imag, N_MC_samples=10000)
    
    # Step 5: display the bands and the Hilbert fitting of the real and the imaginary parts
    
    # Step 5.1: Real part
    
    # Step 5.1.1: Bayesian regression
    entry.mu_Z_re = out_dict_real.get('mu_Z')
    entry.cov_Z_re = np.diag(out_dict_real.get('Sigma_Z'))

    entry.mu_R_inf = out_dict_real.get('mu_gamma')[0]
    entry.cov_R_inf = np.diag(out_dict_real.get('Sigma_gamma'))[0]

    # Step 5.1.2: DRT part
    entry.mu_Z_DRT_re = out_dict_real.get('mu_Z_DRT')
    entry.cov_Z_DRT_re = np.diag(out_dict_real.get('Sigma_Z_DRT'))

    # Step 5.1.3: HT prediction
    entry.mu_Z_H_im = out_dict_real.get('mu_Z_H')
    entry.cov_Z_H_im = np.diag(out_dict_real.get('Sigma_Z_H'))

    # Step 5.1.4: sigma_n estimation
    entry.sigma_n_re = out_dict_real.get('theta')[0]

    # Step 5.1.5: mu_gamma estimation
    entry.mu_gamma_re = out_dict_real.get('mu_gamma')
    entry.out_tau_vec,entry.mu_gamma_fine_re = basics.x_to_gamma(entry.mu_gamma_re[1:],entry.tau_fine,entry.tau, entry.epsilon, rbf_type)
    
    # Step 5.2: Imaginary part
    
    # Step 5.2.1: Bayesian regression
    entry.mu_Z_im = out_dict_imag.get('mu_Z')
    entry.cov_Z_im = np.diag(out_dict_imag.get('Sigma_Z'))

    entry.mu_L_0 = out_dict_imag.get('mu_gamma')[0]
    entry.cov_L_0 = np.diag(out_dict_imag.get('Sigma_gamma'))[0]

    # Step 5.2.2: DRT part
    entry.mu_Z_DRT_im = out_dict_imag.get('mu_Z_DRT')
    entry.cov_Z_DRT_im = np.diag(out_dict_imag.get('Sigma_Z_DRT'))
    
    # Step 5.2.3: HT prediction
    entry.mu_Z_H_re = out_dict_imag.get('mu_Z_H')
    entry.cov_Z_H_re = np.diag(out_dict_imag.get('Sigma_Z_H'))

    # Step 5.2.4: sigma_n estimation
    entry.sigma_n_im = out_dict_imag.get('theta')[0]

    # Step 5.2.5: mu_gamma estimation
    entry.mu_gamma_im = out_dict_imag.get('mu_gamma')
    entry.out_tau_vec,entry.mu_gamma_fine_im = basics.x_to_gamma(entry.mu_gamma_im[1:], entry.tau_fine, entry.tau, entry.epsilon, rbf_type)
    
    # Step 6: plot the fits
    entry.mu_Z_H_re_agm = entry.mu_R_inf + entry.mu_Z_H_re
    entry.band_re_agm = sqrt(entry.cov_R_inf + entry.cov_Z_H_re + entry.sigma_n_im**2)

    entry.mu_Z_H_im_agm = omega_vec*entry.mu_L_0 + entry.mu_Z_H_im
    entry.band_im_agm = sqrt((omega_vec**2)*entry.cov_L_0 + entry.cov_Z_H_im + entry.sigma_n_re**2)

    # Step 7: residuals of the Hilbert DRT
    entry.res_H_re = entry.mu_Z_H_re_agm-entry.b_re
    entry.res_H_im = entry.mu_Z_H_im_agm-entry.b_im
    
    entry.method = 'BHT'    
    
    return entry


## For peak analysis

def peak_analysis(entry, rbf_type = 'Gaussian', data_used = 'Combined Re-Im Data', induct_used = 1, der_used = '1st order', cv_type = 'GCV', shape_control = 'FWHM Coefficient', coeff = 0.5, peak_method = 'separate', N_peaks=1):
     
    """
       This function enables to identify the DRT peaks.
       Inputs:
            entry: an EIS spectrum
            rbf_type: discretization function
            data_used: part of the EIS spectrum used for regularization
            induct_used: treatment of the inductance part
            der_used: order of the derivative considered for the M matrix
            cv_type: regularization method used to select the regularization parameter for ridge regression 
            shape_control: option for controlling the shape of the radial basis function (RBF) 
            coeff: magnitude of the shape control
            N_peaks: desired number of peaks
            peak_method: option for the fit of the recovered DRT, either 'separate' to separately fit each peak, or 'combine' to optimize all the peaks at once
    """
    
    # Step 1: define the necessary quantities before the subsequent optimizations
    
    entry.N_peaks = np.int_(N_peaks)
    simple_run(entry, rbf_type, data_used, induct_used, der_used, cv_type, shape_control, coeff)
    
    # upper and lower log tau values
    log_tau_min = np.min(np.log(entry.out_tau_vec))
    log_tau_max = np.max(np.log(entry.out_tau_vec))

    # list of gaussian matrices
    p_fit_list = [0]*entry.N_peaks  # list of parameter values 
    lb_list = [0]*entry.N_peaks    # list of lower bounds for the parameter values of each peak
    ub_list = [0]*entry.N_peaks    # list of upper bounds for the parameter values of each peak
    
    # list of gaussian matrices
    entry.gamma_fit_list = [0]*entry.N_peaks # separate case
    
    diff_gamma = np.copy(entry.gamma) # difference between the recovered DRT and the DRT with fitted peaks
    
        
    # Step 2: optimal fit of each DRT peak
    
    for n in range(entry.N_peaks):
        
        # bounds for the N_peaks optimizations
        max_sigma_f = 1.3*np.sqrt(np.max(diff_gamma))
        lb = np.array([0, log_tau_min, 1.0/(log_tau_max-log_tau_min)]) # lower bounds for the subsequent minimization
        ub = np.array([max_sigma_f, log_tau_max, np.inf]) # upper bounds for the subsequent minimization
        lb_list[n] = lb
        ub_list[n] = ub
        
        # residual function to minimize for each DRT peak
        residual_fc = lambda p: np.linalg.norm(basics.gauss_fct(p, entry.out_tau_vec, 1) - diff_gamma) ** 2
        
        # initial guesses for the parameter values for each DRT peak
        index_diff_peak = np.argmax(diff_gamma)
        log_tau_diff_peak = np.log(entry.out_tau_vec[index_diff_peak])
        diff_gamma_at_peak = diff_gamma[index_diff_peak]
        sigma_f_at_peak = np.sqrt(diff_gamma_at_peak)
        p_0 = np.array([sigma_f_at_peak, log_tau_diff_peak, 1]) 
        
        # minimization
        options = {'disp': True, 'maxiter': 1e5} # additional options for the optimization
        
        # optimized p values and optimized fit of each DRT peak
        out_fit = minimize(residual_fc, p_0, bounds=list(zip(lb, ub)), method='L-BFGS-B', options=options)
        p_fit = out_fit.x
        gamma_fit = basics.gauss_fct(p_fit, entry.out_tau_vec, 1)
        
        # save the n-th p values
        p_fit_list[n] = p_fit
        
        # save the n-th gamma fit
        entry.gamma_fit_list[n] = gamma_fit
        
        # initial DRT without the peak that was just fitted
        diff_gamma -= gamma_fit
    
    # Step 3: optimal fit of the complete DRT function
    
    # bounds for the overall optimization
    bnds_list = [] # concatenation of all the p values
    
    for k in range(entry.N_peaks):
        bnds_list.extend([(lb_list[k][0], ub_list[k][0]), (lb_list[k][1], ub_list[k][1]), (lb_list[k][2], ub_list[k][2])])     
    
    # initial values of the parameters for the overall optimization
    p_fit_ini = np.concatenate([p_fit_list[k] for k in range(entry.N_peaks)])
    
    # residual function to minimize
    residual_fct_fit = lambda p: np.linalg.norm(basics.gauss_fct(p, entry.out_tau_vec, entry.N_peaks) - entry.gamma) ** 2
    
    # minimization
    out_fit_tot = minimize(residual_fct_fit, p_fit_ini, method='L-BFGS-B', options=options, bounds = bnds_list)
    
    # optimized p values and optimized fit
    p_fit_tot = out_fit_tot.x
    entry.gamma_fit_tot = basics.gauss_fct(p_fit_tot, entry.out_tau_vec, entry.N_peaks)
    
    if peak_method == 'separate': # separate fit of the DRT
    
        entry.out_gamma_fit = entry.gamma_fit_list
    
    else: # combine fit of the DRT
    
        entry.out_gamma_fit = entry.gamma_fit_tot
        
    entry.method = 'peak'   
    
    return entry
