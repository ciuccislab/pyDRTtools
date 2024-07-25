# -*- coding: utf-8 -*-
__authors__ = 'Francesco Ciucci, Ting Hei Wan, Baptiste Py, Adeleke Maradesa'

__date__ = '10th April 2024'


from math import pi, log
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as MVN

"""
Reference: J. Liu, T. H. Wan, F. Ciucci, A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores, Electrochimica Acta. 357 (2020) 136864.
"""

def compute_JSD(mu_P, Sigma_P, mu_Q, Sigma_Q, N_MC_samples): # this function computes the Jensen-Shannon distance (JSD)
    
    out_JSD = np.empty_like(mu_P)

    for index in range(mu_P.size):
        RV_p = MVN(mean=mu_P[index], cov=Sigma_P[index, index])
        RV_q = MVN(mean=mu_Q[index], cov=Sigma_Q[index, index])
    
        x = RV_p.rvs(N_MC_samples)
        p_x = RV_p.pdf(x)
        q_x = RV_q.pdf(x)
        m_x = (p_x+q_x)/2.
    
        y = RV_q.rvs(N_MC_samples)
        p_y = RV_p.pdf(y)
        q_y = RV_q.pdf(y)
        m_y = (p_y+q_y)/2.
    
        dKL_pm = np.log(p_x/m_x).mean()
        dKL_qm = np.log(q_y/m_y).mean()
    
        out_JSD[index] = 0.5*(dKL_pm+dKL_qm) # see (42) in the main manuscript
    
    return out_JSD


def compute_SHD(mu_P, Sigma_P, mu_Q, Sigma_Q): # this function computes the Squared Hellinger distance (SHD)

    sigma_P = np.sqrt(np.diag(Sigma_P))
    sigma_Q = np.sqrt(np.diag(Sigma_Q)) 
    sum_cov = sigma_P**2+sigma_Q**2
    prod_cov = sigma_P*sigma_Q
    out_SHD = 1. - np.sqrt(2.*prod_cov/sum_cov)*np.exp(-0.25*(mu_P-mu_Q)**2/sum_cov) # see (38) in the main manuscript

    return out_SHD


def compute_res_score(res, band): # this function counts the points fallen inside the 1, 2, and 3 sigma credible bands
    
    count = np.zeros(3)
    for k in range(3):
        count[k] = np.sum(np.logical_and(res < (k+1)*band, res > -(k+1)*band))
        
    return count/len(res)

    
def NMLL_fct(theta, Z, A, M, N_freqs, N_taus):
    
    """
    This function computes the hyperparameters in the Bayesian-Hilbert framework
    Inputs:
        theta: hyperparameter vector
        Z: full impedance spactra
        A: matrix obtained by stacking A_re and A_im i.e., A = vstack((A_re,A_im))
        M: derivative matrix
        N_freqs: size of the frequency vector
        N_taus: size of thetimescale vector
    Outputs:
        theta: vector of hyperparameters
    """
    
    sigma_n, sigma_beta, sigma_lambda = theta

    # keep as above
    W = 1/(sigma_beta**2)*np.eye(N_taus+1) + 1/(sigma_lambda**2)*M # see (18) in the main manuscript
    K_agm = 1/(sigma_n**2)*A.T@A + W # see (21b) in the main manuscript

    L_W = np.linalg.cholesky(W)
    L_agm = np.linalg.cholesky(K_agm)

    # compute mu_x
    u = np.linalg.solve(L_agm, A.T@Z)
    u = np.linalg.solve(L_agm.T, u)
    mu_x = 1/(sigma_n**2)*u # see (21a) in the main manuscript

    # compute loss
    E_mu_x = 0.5/(sigma_n**2)*np.linalg.norm(A@mu_x-Z)**2 + 0.5*(mu_x.T@(W@mu_x)) # see (6) in the supplementary information
        
    val_1 = np.sum(np.log(np.diag(L_W)))
    val_2 = - np.sum(np.log(np.diag(L_agm)))
    val_3 = - N_freqs/2.*log(sigma_n**2)
    val_4 = - E_mu_x
    val_5 = - N_freqs/2*log(2*pi)
    
    out_NMLL = -(val_1+val_2+val_3+val_4+val_5) # see (32) in the main manuscript

    return out_NMLL


def HT_single_est(theta_0, Z_exp, A, A_H, M, N_freqs, N_taus):

    """
    This function computes the mean vectors and covariance matrices in the Bayesian-Hilbert framework
    Inputs:
        theta_0: initial guess for the vector of hyperparameters
        Z_exp: experimental data 
        A: matrix obtained by stacking A_re and A_im i.e., A = vstack((A_re,A_im))
        A_H: matrix H (see (26) in the main manuscript)
        M: derivative matrix
        N_freqs: size of the frequency vector
        N_taus: size of thetimescale vector
    Outputs:
        out_dict: dictionaries containing the mean vectors and covariance matrices
    """

    # step 1: identify the vector of hyperparameters
    
    def print_results(theta):
        print('%.5e     %4.6f     %4.6f '%(theta[0], theta[1], theta[2]))

    print('sigma_n; sigma_beta; sigma_lambda')
    res = minimize(NMLL_fct, theta_0, args=(Z_exp, A, M, N_freqs, N_taus), callback=print_results, options={'gtol': 1E-8, 'disp': True})

    # step 2: collect the optimized theta's
    sigma_n, sigma_beta, sigma_lambda = res.x

    # step 3: compute the pdf's of data regression
    # $K_agm = A.T A +\lambda L.T L$
    W = 1/(sigma_beta**2)*np.eye(N_taus+1) + 1/(sigma_lambda**2)*M # see (18) in the main manuscript
    K_agm = 1/(sigma_n**2)*A.T@A + W # see (21b) in the main manuscript

    # step 4: compute the Cholesky factorization to obtain the inverse of the covariance matrix
    L_agm = np.linalg.cholesky(K_agm)
    inv_L_agm = np.linalg.inv(L_agm)
    inv_K_agm = inv_L_agm.T@inv_L_agm

    # step 5: compute the gamma ~ N(mu_gamma, Sigma_gamma)
    Sigma_gamma = inv_K_agm
    mu_gamma = 1/(sigma_n**2)*(Sigma_gamma@A.T)@Z_exp.real # see (21a) in the main manuscript
    
    # step 6: compute, from gamma, the Z ~ N(mu_Z, Sigma_Z)
    mu_Z = A@mu_gamma 
    Sigma_Z = A@(Sigma_gamma@A.T) + sigma_n**2*np.eye(N_freqs) 
    
    # step 7: compute, from gamma, the Z_DRT ~ N(mu_Z_DRT, Sigma_Z_DRT)
    A_DRT = A[:,1:]
    mu_gamma_DRT = mu_gamma[1:]
    Sigma_gamma_DRT = Sigma_gamma[1:,1:]
    mu_Z_DRT = A_DRT@mu_gamma_DRT # see (30a) in the main manuscript
    Sigma_Z_DRT = A_DRT@(Sigma_gamma_DRT@A_DRT.T) # see (30b) in the main manuscript
    
    # step 8: compute, from gamma, the Z_H_conj ~ N(mu_Z_H_conj, Sigma_Z_H_conj)   
    mu_Z_H = A_H@mu_gamma[1:] # see (25a) in the main manuscript
    Sigma_Z_H = A_H@(Sigma_gamma[1:,1:]@A_H.T) # see (25b) in the main manuscript

    out_dict = {
        'mu_gamma': mu_gamma,
        'Sigma_gamma': Sigma_gamma,
        'mu_Z': mu_Z,
        'Sigma_Z': Sigma_Z,
        'mu_Z_DRT': mu_Z_DRT,
        'Sigma_Z_DRT': Sigma_Z_DRT,        
        'mu_Z_H': mu_Z_H,
        'Sigma_Z_H': Sigma_Z_H,
        'theta': res.x
    }
    return out_dict


def EIS_score(theta_0, freq_vec, Z_exp, out_dict_real, out_dict_imag, N_MC_samples=10000):
    
    """
    This function computes various scores to ascertain the quality of the DRT and impedance recovery
    Inputs:
        theta_0: initial guess for the vector of hyperparameters
        freq_vec: frequency vector
        Z_exp: experimental data 
        out_dict_real: real part of the dictionaries outputted by HT_single_est
        out_dict_imag: imaginary part of the dictionaries outputted by HT_single_est
        N_MC_samples: number of samples
    Outputs:
        dictionary containing the scores
     """
     
    # s_mu - distance between means:
    mu_Z_DRT_re = out_dict_real.get('mu_Z_DRT')
    mu_Z_DRT_im = out_dict_imag.get('mu_Z_DRT')
    mu_Z_H_re = out_dict_imag.get('mu_Z_H')
    mu_Z_H_im = out_dict_real.get('mu_Z_H')

    discrepancy_re = np.linalg.norm(mu_Z_DRT_re-mu_Z_H_re)/(np.linalg.norm(mu_Z_DRT_re)+np.linalg.norm(mu_Z_H_re))
    s_mu_re = 1. - discrepancy_re # see (36a) in the main manuscript
    discrepancy_im = np.linalg.norm(mu_Z_DRT_im-mu_Z_H_im)/(np.linalg.norm(mu_Z_DRT_im)+np.linalg.norm(mu_Z_H_im))
    s_mu_im = 1. - discrepancy_im # see (36b) in the main manuscript
    
    # s_JSD - Jensen-Shannon Distance:
    # we need the means (above) and covariances (below) for the computation of the JSD
    Sigma_Z_DRT_re = out_dict_real.get('Sigma_Z_DRT')
    Sigma_Z_DRT_im = out_dict_imag.get('Sigma_Z_DRT')
    Sigma_Z_H_re = out_dict_imag.get('Sigma_Z_H')
    Sigma_Z_H_im = out_dict_real.get('Sigma_Z_H')

    # s_res - residual score:
    # real part
    # retrieve distribution of R_inf
    mu_R_inf = out_dict_real.get('mu_gamma')[0]
    cov_R_inf = np.diag(out_dict_real.get('Sigma_gamma'))[0]
    # we will also need omega and an estimate of the error
    sigma_n_im = out_dict_imag.get('theta')[0]
    
    # R_inf+Z_H_re-Z_exp has:
    # mean:
    res_re = mu_R_inf + mu_Z_H_re - Z_exp.real
    # std:
    band_re = np.sqrt(cov_R_inf + np.diag(Sigma_Z_H_re)+sigma_n_im**2)
    s_res_re = compute_res_score(res_re, band_re)
    
    # imaginary part
    # retrieve distribution of L_0     
    mu_L_0 = out_dict_imag.get('mu_gamma')[0]
    cov_L_0 = np.diag(out_dict_imag.get('Sigma_gamma'))[0]
    # we will also need omega
    omega_vec = 2.*pi*freq_vec
    # and an estimate of the error
    sigma_n_re = out_dict_real.get('theta')[0]

    # R_inf+Z_H_re-Z_exp has:

    # mean:
    res_im = omega_vec*mu_L_0 + mu_Z_H_im - Z_exp.imag 
    # std:
    band_im = np.sqrt((omega_vec**2)*cov_L_0 + np.diag(Sigma_Z_H_im)+sigma_n_re**2)
    s_res_im = compute_res_score(res_im, band_im)

    # Squared Hellinger distance (SHD)
    # which is bounded between 0 and 1
    SHD_re = compute_SHD(mu_Z_DRT_re, Sigma_Z_DRT_re, mu_Z_H_re, Sigma_Z_H_re)
    SHD_im = compute_SHD(mu_Z_DRT_im, Sigma_Z_DRT_im, mu_Z_H_im, Sigma_Z_H_im)

    # we are going to score w.r.t. the Hellinger distance (HD)
    # the score uses 1 to mean good (this means close)
    # and 0 means bad (far away) => that's the opposite of the distance
    s_HD_re = 1.-np.sqrt(SHD_re).mean() # see (40a) in the main manuscript
    s_HD_im = 1.-np.sqrt(SHD_im).mean() # see (40b) in the main manuscript

    # compute the Jensen-Shannon distance (JSD)
    JSD_re = compute_JSD(mu_Z_DRT_re, Sigma_Z_DRT_re, mu_Z_H_re, Sigma_Z_H_re, N_MC_samples)
    JSD_im = compute_JSD(mu_Z_DRT_im, Sigma_Z_DRT_im, mu_Z_H_im, Sigma_Z_H_im, N_MC_samples)

    # the JSD is a symmetrized relative entropy (discrepancy), so highest value means more entropy
    # we are going to reverse that by taking (log(2)-JSD)/log(2)
    # which means higher value less relative entropy (discrepancy)
    s_JSD_re = (log(2)-JSD_re.mean())/log(2) # see (44a) in the main manuscript
    s_JSD_im = (log(2)-JSD_im.mean())/log(2) # see (44b) in the main manuscript

    out_scores = {
        's_res_re': s_res_re,
        's_res_im': s_res_im,
        's_mu_re': s_mu_re,
        's_mu_im': s_mu_im,
        's_HD_re': s_HD_re,
        's_HD_im': s_HD_im,
        's_JSD_re': s_JSD_re,
        's_JSD_im': s_JSD_im
    }
    
    return out_scores