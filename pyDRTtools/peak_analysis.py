__authors__ = 'Francesco Ciucci, Adeleke Maradesa, Baptiste Py, Ting Hei Wan'

__date__ = '11th April 2024'

import numpy as np

# this code performs peak deconvolution using either Gaussian, Havriliak-Negami (HN), or ZARC functions

def peak_fct(p, tau_vec, N_peaks, fit='Gaussian'):
    
    """
    This function returns a fit of the peaks in the DRT spectrum

    Inputs:
        p: parameters of the Gaussian functions (sigma_f, mu_log_tau, and inv_sigma for each DRT peak)
        tau_vec: vector of timescales
        N_peaks: number of peaks in the DRT spectrum
        fit: nature of the DRT fit (Gaussian, HN, or ZARC)

    Output:
        gamma_out: sum of Gaussian functions
    """
    
    if fit=='Gaussian': # fit with Gaussian functions
        
        gamma_out = np.zeros_like(tau_vec) # sum of Gaussian functions, whose parameters (the prefactor sigma_f, mean mu_log_tau, and standard deviation 1/inv_sigma for each DRT peak) are encapsulated in p
    
        for k in range(N_peaks):
        
            sigma_f, mu_log_tau, inv_sigma = p[3*k:3*k+3] 
            gaussian_out = sigma_f**2*np.exp(-inv_sigma**2/2*((np.log(tau_vec) - mu_log_tau)**2)) # we use inv_sigma because this leads to less computational problems (no exploding gradient when sigma->0)
            gamma_out += gaussian_out 
            
    elif fit=='Havriliak-Negami': # fit with HN functions
        
        gamma_out = np.zeros_like(tau_vec) # sum of single-ZARC functions, whose parameters (R_ct, log_tau_0, phi for each DRT peak) are encapsulated in p
        
        for k in range(N_peaks):
            
            R_ct, log_tau_0, phi, psi = p[4*k:4*k+4] 
            
            x = np.exp(phi*(np.log(tau_vec)-log_tau_0))
            
            theta = np.arctan(np.abs(np.sin(np.pi*phi)/(x+np.cos(np.pi*phi))))
            
            num = R_ct*x**psi*np.sin(psi*theta)
            denom = np.pi*(1+np.cos(np.pi*phi)*x+x**2)**(psi/2)
            
            DRT_HN_out = num/denom
            
            gamma_out += DRT_HN_out 
            
    else: # fit with ZARC functions
    
        gamma_out = np.zeros_like(tau_vec) # sum of single-ZARC functions, whose parameters (R_ct, log_tau_0, phi for each DRT peak) are encapsulated in p
        
        for k in range(N_peaks):
            
            R_ct, log_tau_0, phi = p[3*k:3*k+3] 
            x = np.exp(phi*(np.log(tau_vec)-log_tau_0))
            DRT_ZARC_out = R_ct*np.sin(np.pi*phi)*x/(1+2*np.cos(np.pi*phi)*x+x**2)
            gamma_out += DRT_ZARC_out 
            
    return gamma_out
