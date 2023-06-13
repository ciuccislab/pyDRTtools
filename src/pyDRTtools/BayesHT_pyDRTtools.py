# -*- coding: utf-8 -*-
__authors__ = 'Francesco Ciucci, Ting Hei Wan'

__date__ = '12th June 2023'


from math import pi, log, sqrt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as MVN


"""
Reference: J. Liu, T. H. Wan, F. Ciucci, A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores, Electrochimica Acta. 357 (2020) 136864.
"""


def compute_A_re(freq_vec, tau_vec, flag='impedance'):
    
    """
    This function computes the approximation matrix of the DRT for the real part of the EIS data
    Inputs:
        freq_vec: frequency vector
        tau_vec: timescale vector
        flag: nature of the data, i.e., impedance or admittance
    Output:
        approximation matrix for the real part of the impedance
    """
    
    omega_vec = 2.*pi*freq_vec
    N_freqs = freq_vec.size
    N_taus = tau_vec.size

    out_A_re = np.zeros((N_freqs, N_taus+1))
    out_A_re[:,0] = 1.
    
    if flag == 'impedance': # see equation (11a) in the main manuscript
        for p in range(0, N_freqs):
            for q in range(0, N_taus):
                if q == 0:
                    out_A_re[p, q+1] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                elif q == N_taus-1:
                    out_A_re[p, q+1] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                else:
                    out_A_re[p, q+1] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])
    
    # for the admittance calculations
    else:  # see equation (16a) in the supplementary information
        for p in range(0, N_freqs):
            for q in range(0, N_taus):
                if q == 0:
                    out_A_re[p, q+1] = 0.5*(omega_vec[p]**2*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                elif q == N_taus-1:
                    out_A_re[p, q+1] = 0.5*(omega_vec[p]**2*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                else:
                    out_A_re[p, q+1] = 0.5*(omega_vec[p]**2*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])

    return out_A_re


def compute_A_H_re(freq_vec, tau_vec, flag):
    
    out_A_re = compute_A_re(freq_vec, tau_vec, flag)
    out_A_H_re = out_A_re[:,1:]
    
    return out_A_H_re


def compute_A_im(freq_vec, tau_vec, flag='impedance'):
    
    """
    This function computes the approximation matrix of the DRT for the imaginary part of the EIS data
    Inputs:
        freq_vec: frequency vector
        tau_vec: timescale vector
        flag: nature of the data, i.e., impedance or admittance
    Output:
        approximation matrix for the imaginary part of the impedance A_im
    """
    
    omega_vec = 2.*pi*freq_vec

    N_taus = tau_vec.size
    N_freqs = freq_vec.size

    out_A_im = np.zeros((N_freqs, N_taus+1))
    out_A_im[:,0] = omega_vec

    if flag == 'impedance': # see equation (11b) in the main manuscript
        for p in range(0, N_freqs):
            for q in range(0, N_taus):
                if q == 0:
                    out_A_im[p, q+1] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                elif q == N_taus-1:
                    out_A_im[p, q+1] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                else:
                    out_A_im[p, q+1] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])
    
    # for the admittance calculations
    else: # see equation (16b) in the supplementary information
        for p in range(0, N_freqs):
            for q in range(0, N_taus):
                if q == 0:
                    out_A_im[p, q+1] = 0.5*(omega_vec[p])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                elif q == N_taus-1:
                    out_A_im[p, q+1] = 0.5*(omega_vec[p])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                else:
                    out_A_im[p, q+1] = 0.5*(omega_vec[p])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])

    return out_A_im


def compute_A_H_im(freq_vec, tau_vec, flag='impedance'):
    
    out_A_im = compute_A_im(freq_vec, tau_vec, flag)
    out_A_H_im = out_A_im[:,1:]
    
    return out_A_H_im


def compute_L1(tau_vec):
    
    # This function computes the differenciation matrix (L1) using first-order finite differencing (see (18) in the main manuscript)
    
    N_taus = tau_vec.size
    out_L1 = np.zeros((N_taus-2, N_taus+1))
    
    for p in range(0, N_taus-2):

        delta_loc = log(tau_vec[p+1]/tau_vec[p])
        
        if p==0:
            out_L1[p,p+1] = -3./(2*delta_loc)
            out_L1[p,p+2] = 4./(2*delta_loc)
            out_L1[p,p+3] = -1./(2*delta_loc)
        elif p == N_taus-2:
            out_L1[p,p]   = 1./(2*delta_loc)
            out_L1[p,p+1] = -4./(2*delta_loc)
            out_L1[p,p+2] = 3./(2*delta_loc)
        else:
            out_L1[p,p] = 1./(2*delta_loc)
            out_L1[p,p+2] = -1./(2*delta_loc)
            
    return out_L1


def compute_L2(tau_vec):
    
    # this function computes the differenciation matrix (L2) using second-order finite differencing (see (18) in the main manuscript)
    
    N_taus = tau_vec.size
    out_L2 = np.zeros((N_taus-2, N_taus+1))
    
    for p in range(0, N_taus-2):

        delta_loc = log(tau_vec[p+1]/tau_vec[p])
        
        if p==0 or p == N_taus-3:
            out_L2[p,p+1] = 2./(delta_loc**2)
            out_L2[p,p+2] = -4./(delta_loc**2)
            out_L2[p,p+3] = 2./(delta_loc**2)
            
        else:
            out_L2[p,p+1] = 1./(delta_loc**2)
            out_L2[p,p+2] = -2./(delta_loc**2)
            out_L2[p,p+3] = 1./(delta_loc**2)
            
    return out_L2


def compute_JSD(mu_P, Sigma_P, mu_Q, Sigma_Q, N_MC_samples):

    # This function computes the Jensen-Shannon distance (JSD)
    
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
    
        out_JSD[index] = 0.5*(dKL_pm+dKL_qm) # see equation (42) in the main manuscript
    
    return out_JSD


def compute_SHD(mu_P, Sigma_P, mu_Q, Sigma_Q):

    # This function computes the Squared Hellinger distance (SHD)

    sigma_P = np.sqrt(np.diag(Sigma_P))
    sigma_Q = np.sqrt(np.diag(Sigma_Q)) 
    sum_cov = sigma_P**2+sigma_Q**2
    prod_cov = sigma_P*sigma_Q
    out_SHD = 1. - np.sqrt(2.*prod_cov/sum_cov)*np.exp(-0.25*(mu_P-mu_Q)**2/sum_cov) # see equation (38) in the main manuscript

    return out_SHD


def compute_res_score(res, band):

    # This function counts the points fallen inside the 1, 2, and 3 sigma credible bands
    
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
    W = 1/(sigma_beta**2)*np.eye(N_taus+1) + 1/(sigma_lambda**2)*M # see equation (18) in the main manuscript
    K_agm = 1/(sigma_n**2)*A.T@A + W # see equation (21b) in the main manuscript

    L_W = np.linalg.cholesky(W)
    L_agm = np.linalg.cholesky(K_agm)

    # compute mu_x
    u = np.linalg.solve(L_agm, A.T@Z)
    u = np.linalg.solve(L_agm.T, u)
    mu_x = 1/(sigma_n**2)*u # see equation (21a) in the main manuscript

    # compute loss
    E_mu_x = 0.5/(sigma_n**2)*np.linalg.norm(A@mu_x-Z)**2 + 0.5*(mu_x.T@(W@mu_x)) # see equation (6) in the supplementary information
        
    val_1 = np.sum(np.log(np.diag(L_W)))
    val_2 = - np.sum(np.log(np.diag(L_agm)))
    val_3 = - N_freqs/2.*log(sigma_n**2)
    val_4 = - E_mu_x
    val_5 = - N_freqs/2*log(2*pi)
    
    out_NMLL = -(val_1+val_2+val_3+val_4+val_5) # see equation (32) in the main manuscript

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
    # theta_0_refined = res.x
    # res = minimize(NMLL_fct, theta_0_refined, args=(Z_exp, A, L, N_freqs, N_taus), callback=print_results, options={'disp': True}, method = 'Nelder-Mead', tol = 1E-8)

    # step 2: collect the optimized theta's
    sigma_n, sigma_beta, sigma_lambda = res.x

    # step 3: compute the pdf's of data regression
    # $K_agm = A.T A +\lambda L.T L$
    W = 1/(sigma_beta**2)*np.eye(N_taus+1) + 1/(sigma_lambda**2)*M # see equation (18) in the main manuscript
    K_agm = 1/(sigma_n**2)*A.T@A + W # see equation (21b) in the main manuscript

    # step 4: compute the Cholesky factorization to obtain the inverse of the covariance matrix
    L_agm = np.linalg.cholesky(K_agm)
    inv_L_agm = np.linalg.inv(L_agm)
    inv_K_agm = inv_L_agm.T@inv_L_agm

    # step 5: compute the gamma ~ N(mu_gamma, Sigma_gamma)
    Sigma_gamma = inv_K_agm
    mu_gamma = 1/(sigma_n**2)*(Sigma_gamma@A.T)@Z_exp.real # see equation (21a) in the main manuscript
    
    # step 6: compute, from gamma, the Z ~ N(mu_Z, Sigma_Z)
    mu_Z = A@mu_gamma 
    Sigma_Z = A@(Sigma_gamma@A.T) + sigma_n**2*np.eye(N_freqs) 
    
    # step 7: compute, from gamma, the Z_DRT ~ N(mu_Z_DRT, Sigma_Z_DRT)
    A_DRT = A[:,1:]
    mu_gamma_DRT = mu_gamma[1:]
    Sigma_gamma_DRT = Sigma_gamma[1:,1:]
    mu_Z_DRT = A_DRT@mu_gamma_DRT # see equation (30a) in the main manuscript
    Sigma_Z_DRT = A_DRT@(Sigma_gamma_DRT@A_DRT.T) # see equation (30b) in the main manuscript
    
    # step 8: compute, from gamma, the Z_H_conj ~ N(mu_Z_H_conj, Sigma_Z_H_conj)   
    mu_Z_H = A_H@mu_gamma[1:] # see equation (25a) in the main manuscript
    Sigma_Z_H = A_H@(Sigma_gamma[1:,1:]@A_H.T) # see equation (25b) in the main manuscript

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
    s_mu_re = 1. - discrepancy_re # see equation (36a) in the main manuscript
    discrepancy_im = np.linalg.norm(mu_Z_DRT_im-mu_Z_H_im)/(np.linalg.norm(mu_Z_DRT_im)+np.linalg.norm(mu_Z_H_im))
    s_mu_im = 1. - discrepancy_im # see equation (36b) in the main manuscript
    
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
    s_HD_re = 1.-np.sqrt(SHD_re).mean() # see equation (40a) in the main manuscript
    s_HD_im = 1.-np.sqrt(SHD_im).mean() # see equation (40b) in the main manuscript

    # compute the Jensen-Shannon distance (JSD)
    JSD_re = compute_JSD(mu_Z_DRT_re, Sigma_Z_DRT_re, mu_Z_H_re, Sigma_Z_H_re, N_MC_samples)
    JSD_im = compute_JSD(mu_Z_DRT_im, Sigma_Z_DRT_im, mu_Z_H_im, Sigma_Z_H_im, N_MC_samples)

    # the JSD is a symmetrized relative entropy (discrepancy), so highest value means more entropy
    # we are going to reverse that by taking (log(2)-JSD)/log(2)
    # which means higher value less relative entropy (discrepancy)
    s_JSD_re = (log(2)-JSD_re.mean())/log(2) # see equation (44a) in the main manuscript
    s_JSD_im = (log(2)-JSD_im.mean())/log(2) # see equation (44b) in the main manuscript

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

    
def HT_est(theta_0, Z_exp, freq_vec, tau_vec, Dn='D2', data_flag='impedance'):
    
    """
    This function computes the mean vectors, covariance matrices, and scores 
    Inputs:
        theta_0: initial guess for the vector of hyperparameters
        Z_exp: experimental data 
        freq_vec: frequency vector
        tau_vec: timescale vector
        Dn: order of the differenciation matrix
        flag: nature of the data, i.e., impedance or admittance
    Outputs:
        dictionary containing the mean vectors, covariance matrices, and scores
     """
    
    N_freqs = freq_vec.size
    N_taus = tau_vec.size

    # compute the matrix $A = A_re + i A_im$
    A_re = compute_A_re(freq_vec, tau_vec, flag=data_flag)
    A_im = compute_A_im(freq_vec, tau_vec, flag=data_flag)
    # as well as the ones used for the Hilbert transform
    A_H_re = compute_A_H_re(freq_vec, tau_vec, flag=data_flag)
    A_H_im = compute_A_H_im(freq_vec, tau_vec, flag=data_flag)

    # compute the matrix L
    if Dn == 'D1':
        L = compute_L1(tau_vec)
    else:
        L = compute_L2(tau_vec)

    # estimates
    out_dict_real = HT_single_est(theta_0, Z_exp.real, A_re, A_H_im, L, N_freqs, N_taus)
    out_dict_imag = HT_single_est(theta_0, Z_exp.imag, A_im, A_H_re, L, N_freqs, N_taus)
    out_scores = EIS_score(theta_0, freq_vec, Z_exp, out_dict_real, out_dict_imag)

    return out_dict_real, out_dict_imag, out_scores