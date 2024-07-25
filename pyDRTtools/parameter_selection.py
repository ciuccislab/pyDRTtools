
__authors__ = 'Francesco Ciucci, Adeleke Maradesa, Baptiste Py, Ting Hei Wan, Mohammed B. Effat'

__date__ = '10th April, 2024'

import numpy as np
import sys
import cvxopt
from numpy import exp
from math import log
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from numpy.linalg import norm, cholesky
from numpy import *
from . import basics
from . import nearest_PD as nPD
#import basics
print(sys.path)

"""
References:
    [1] A. Maradesa, B. Py, T.H. Wan, M.B. Effat, F. Ciucci, Selecting the regularization parameter in the 
    distribution of relaxation times, Journal of the Electrochemical Society. 170 (2023) 030502.
    [2] M. Saccoccio, T.H. Wan, C. Chen, F. Ciucci, Optimal regularization in distribution of relaxation 
    times applied to electrochemical impedance spectroscopy: Ridge and lasso regression methods - A 
    theoretical and experimental study, Electrochimica Acta. 147 (2014) 470-482.
"""


def compute_GCV(log_lambda, A_re, A_im, Z_re, Z_im, M, data_used, induct_used):
    
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
           data_used: part of the EIS spectrum used for regularization
           induct_used: treatment of the inductance part
       Output:
           GCV score
    """
    #
    lambda_value = exp(log_lambda)
    
    A = np.vstack((A_re, A_im)) # matrix A with A_re and A_im ; see (5) in [1]
    Z = np.hstack((Z_re, Z_im)) # stacked impedance
    
    n_cv = Z.shape[0] # n_cv = 2*N_freqs with N_freqs the number of EIS frequencies
    
    A_in = A.T@A + lambda_value*M # see (13) in [1]
    
    # check if A_in is positive-definite    
    if nPD.is_PD(A_in)==False: 
        A_in = nPD.nearest_PD(A_in) 
    
    # Cholesky transform of A_in        
    L_in = cholesky(A_in) 
    
    # inverse of A_in based on the Cholesky transform of A_in
    inv_L_in = np.linalg.inv(L_in)
    inv_A_in = inv_L_in.T@inv_L_in # inverse of A_in
    K = A@inv_A_in@A.T  # see (13) in [1]
    
    # GCV score; see (13) in [1]
    GCV_num = 1/n_cv*norm((np.eye(n_cv)-K)@Z)**2 # numerator
    GCV_dom = (1/n_cv*np.trace(np.eye(n_cv)-K))**2 # denominator
    
    GCV_score = GCV_num/GCV_dom
    
    return GCV_score


def compute_mGCV(log_lambda, A_re, A_im, Z_re, Z_im, M, data_used, induct_used):
    
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
           data_used: part of the EIS spectrum used for regularization
           induct_used: treatment of the inductance part
       Output:
           mGCV score
    """
    # 
    lambda_value = exp(log_lambda)
    
    A = np.vstack((A_re, A_im)) # see (5) in [1]
    Z = np.hstack((Z_re, Z_im))
    
    n_cv = Z.shape[0] # 2*number of frequencies
    
    A_in = A.T@A + lambda_value*M # see (13) in [1]
    
    # check if A_in is positive-definite    
    if nPD.is_PD(A_in)==False: 
        A_in = nPD.nearest_PD(A_in) 
    
    # Cholesky transform of A_in        
    L_in = cholesky(A_in) 
    
    # inverse of A_in based on the Cholesky transform of A_in
    inv_L_in = np.linalg.inv(L_in)
    inv_A_in = inv_L_in.T@inv_L_in # inverse of A_in
    K = A@inv_A_in@A.T  # see (13) in [1]
    
    # the stabilization parameter, rho, is computed as described in (15) in [1]
    if n_cv < 50:
        rho = 1.3
    else:
        rho = 2
    
    # mGCV score ; see (14) in [1]
    mGCV_num = 1/n_cv*norm((np.eye(n_cv)-K)@Z)**2 # numerator
    mGCV_dom = ((1/n_cv)*(np.trace(np.eye(n_cv)-rho*K)))**2 # denominator
    mGCV_score = mGCV_num/mGCV_dom
    
    return mGCV_score


def compute_rGCV(log_lambda, A_re, A_im, Z_re, Z_im, M, data_used, induct_used):
    
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
           data_used: part of the EIS spectrum used for regularization
           induct_used: treatment of the inductance part
       Output:
           rGCV score    
    """
    # 
    lambda_value = exp(log_lambda)
    
    A = np.vstack((A_re, A_im)) # see (5) in [1]
    Z = np.hstack((Z_re, Z_im))
    
    n_cv = Z.shape[0] # 2*number of frequencies
    
    A_in = A.T@A + lambda_value*M # see (13) in [1]
    
    # check if A_in is positive-definite    
    if nPD.is_PD(A_in)==False: 
        A_in = nPD.nearest_PD(A_in) 
    
    # Cholesky transform of A_in        
    L_in = cholesky(A_in) 
    
    # inverse of A_in based on the Cholesky transform of A_in
    inv_L_in = np.linalg.inv(L_in)
    inv_A_in = inv_L_in.T@inv_L_in # inverse of A_in
    K = A@inv_A_in@A.T  # see (13) in [1]
    
    # GCV score ; see (13) in [1]
    rGCV_num = 1/n_cv*norm((np.eye(n_cv)-K)@Z)**2
    rGCV_dom = ((1/n_cv)*(np.trace(np.eye(n_cv)-K)))**2
    rGCV = rGCV_num/rGCV_dom
    
    # the robust parameter, xsi, is computed as described in (16) in [1]
    if n_cv<100:
        xi = 0.2
    else:
        xi = 0.3
    
    # mu_2 parameter ; see (16) in [1]
    mu_2 = (1/n_cv)*np.trace(K.T@K)
    
    # rGCV score ; see (16) in [1]
    rGCV_score = (xi + (1-xi)*mu_2)*rGCV
        
    return rGCV_score


def compute_re_im_cv(log_lambda, A_re, A_im, Z_re, Z_im, M, data_used, induct_used):
    
    """
    This function computes the re-im score using CVXOPT to minimize the quadratic problem.
    Inputs:
        log_lambda: regularization parameter
        A_re: discretization matrix for the real part of the impedance
        A_im: discretization matrix for the imaginary part of the impedance
        Z_re: vector of the real parts of the impedance
        Z_im: vector of the imaginary parts of the impedance
        M: differentiation matrix
        data_used: part of the EIS spectrum used for regularization
        induct_used: treatment of the inductance part
    Output:
        re-im score
    """
    
    lambda_value = exp(log_lambda)

    # Obtain H and c matrices for both real and imaginary part
    H_re, c_re = basics.quad_format_separate(A_re, Z_re, M, lambda_value)
    H_im, c_im = basics.quad_format_separate(A_im, Z_im, M, lambda_value)
    
    if data_used == 'Combined Re-Im Data': # select both parts of the impedance for the simple run
 
        if induct_used == 0 or induct_used == 2: # without considering the inductance
            N_RL = 1 # N_RL length of resistance plus inductance
            lb = np.zeros([Z_re.shape[0] + N_RL])  # to include a resistance in the DRT model
            bound_mat = np.eye(lb.shape[0])

            args_re = [cvxopt.matrix(H_re),cvxopt.matrix(c_re), cvxopt.matrix(-bound_mat), cvxopt.matrix(lb)]
            args_im = [cvxopt.matrix(H_im), cvxopt.matrix(c_im), cvxopt.matrix(-bound_mat), cvxopt.matrix(lb)]

            # solve the quadratic programming problems
            sol_re = cvxopt.solvers.qp(*args_re)
            sol_im = cvxopt.solvers.qp(*args_im)

            if 'optimal' not in sol_re['status'] or 'optimal' not in sol_im['status']:
                return None
            
            # obtain gamma vector for real and imaginary parts of the impedance
            gamma_ridge_re = np.array(sol_re['x']).flatten()
            gamma_ridge_im = np.array(sol_im['x']).flatten()

            # stack the resistance R on top of gamma_ridge_im
            gamma_ridge_re_cv = np.hstack((np.array([gamma_ridge_re[0]]), gamma_ridge_im[1:]))
            gamma_ridge_im_cv = np.hstack((np.array([0]), gamma_ridge_re[1:]))
                       
        elif induct_used == 1: # considering the inductance
            N_RL = 2
            lb = np.zeros([Z_re.shape[0] + N_RL])  # to include a resistance and an inductance in the DRT model
            bound_mat = np.eye(lb.shape[0])

            args_re = [cvxopt.matrix(H_re),cvxopt.matrix(c_re), cvxopt.matrix(-bound_mat), cvxopt.matrix(lb)]
            args_im = [cvxopt.matrix(H_im), cvxopt.matrix(c_im), cvxopt.matrix(-bound_mat), cvxopt.matrix(lb)]

            # solve the quadratic programming problems
            sol_re = cvxopt.solvers.qp(*args_re)
            sol_im = cvxopt.solvers.qp(*args_im)

            if 'optimal' not in sol_re['status'] or 'optimal' not in sol_im['status']:
                return None
            
            # obtain gamma vector for real and imaginary parts of the impedance
            gamma_ridge_re = np.array(sol_re['x']).flatten()
            gamma_ridge_im = np.array(sol_im['x']).flatten()

            # stack the resistance R and inductance L on top of gamma_ridge_im and gamma_ridge_re, respectively
            gamma_ridge_re_cv = np.hstack((np.array([0, gamma_ridge_re[1]]), gamma_ridge_im[2:]))
            gamma_ridge_im_cv = np.hstack((np.array([gamma_ridge_im[0], 0]), gamma_ridge_re[2:]))       
        
    elif data_used == 'Im Data': # select the imaginary part of the impedance for the simple run
            
        if induct_used == 0 or induct_used == 2: # without considering the inductance
            N_RL = 0 
            lb = np.zeros([Z_re.shape[0] + N_RL]) # no additional resistance nor inductance in the DRT model
            bound_mat = np.eye(lb.shape[0])

            args_re = [cvxopt.matrix(H_re),cvxopt.matrix(c_re), cvxopt.matrix(-bound_mat), cvxopt.matrix(lb)]
            args_im = [cvxopt.matrix(H_im), cvxopt.matrix(c_im), cvxopt.matrix(-bound_mat), cvxopt.matrix(lb)]

            # solve the quadratic programming problems
            sol_re = cvxopt.solvers.qp(*args_re)
            sol_im = cvxopt.solvers.qp(*args_im)

            if 'optimal' not in sol_re['status'] or 'optimal' not in sol_im['status']:
                return None
            
            # obtain gamma vector for real and imaginary parts of the impedance
            gamma_ridge_re_cv = np.array(sol_im['x']).flatten()
            gamma_ridge_im_cv = np.array(sol_re['x']).flatten()
            
            
        elif induct_used == 1: # considering the inductance
            N_RL = 1
            lb = np.zeros([Z_re.shape[0] + N_RL])  # to include an inductance in the DRT model
            bound_mat = np.eye(lb.shape[0])

            args_re = [cvxopt.matrix(H_re),cvxopt.matrix(c_re), cvxopt.matrix(-bound_mat), cvxopt.matrix(lb)]
            args_im = [cvxopt.matrix(H_im), cvxopt.matrix(c_im), cvxopt.matrix(-bound_mat), cvxopt.matrix(lb)]

            # solve the quadratic programming problems
            sol_re = cvxopt.solvers.qp(*args_re)
            sol_im = cvxopt.solvers.qp(*args_im)

            if 'optimal' not in sol_re['status'] or 'optimal' not in sol_im['status']:
                return None
            
            # obtain gamma vector for real and imaginary parts of the impedance
            gamma_ridge_re = np.array(sol_re['x']).flatten()
            gamma_ridge_im = np.array(sol_im['x']).flatten()

            # stack the inductance L on top of gamma_ridge_re
            gamma_ridge_re_cv = np.hstack((np.array([0]), gamma_ridge_im[1:]))
            gamma_ridge_im_cv = np.hstack((np.array([gamma_ridge_im[0]]), gamma_ridge_re[1:]))

    elif data_used == 'Re Data': # select the real part of the impedance for the simple run
        N_RL = 1
        lb = np.zeros([Z_re.shape[0] + N_RL])  # to include a resistance in the DRT model
        bound_mat = np.eye(lb.shape[0])

        args_re = [cvxopt.matrix(H_re),cvxopt.matrix(c_re), cvxopt.matrix(-bound_mat), cvxopt.matrix(lb)]
        args_im = [cvxopt.matrix(H_im), cvxopt.matrix(c_im), cvxopt.matrix(-bound_mat), cvxopt.matrix(lb)]

        # solve the quadratic programming problems
        sol_re = cvxopt.solvers.qp(*args_re)
        sol_im = cvxopt.solvers.qp(*args_im)

        if 'optimal' not in sol_re['status'] or 'optimal' not in sol_im['status']:
            return None
        
        # obtain gamma vector for real and imaginary parts of the impedance
        gamma_ridge_re = np.array(sol_re['x']).flatten()
        gamma_ridge_im = np.array(sol_im['x']).flatten()

        # stack the resistance R on top of gamma_ridge_im
        gamma_ridge_re_cv = np.hstack((np.array([gamma_ridge_re[0]]), gamma_ridge_im[1:]))
        gamma_ridge_im_cv = np.hstack((np.array([0]), gamma_ridge_re[1:]))

    
    # Re-im score; see (17) in [1] and (13) in [2]
    re_im_cv_score = norm(Z_re - A_re @ gamma_ridge_re_cv) ** 2 + norm(Z_im - A_im @ gamma_ridge_im_cv) ** 2

    return re_im_cv_score


def compute_kf_cv(log_lambda, A_re, A_im, Z_re, Z_im, M, data_used, induct_used):
    
    """
       This function computes the k-fold (kf) score.
       Inputs: 
           log_lambda: regularization parameter
           A_re: discretization matrix for the real part of the impedance
           A_im: discretization matrix for the real part of the impedance
           Z_re: vector of the real parts of the impedance
           Z_im: vector of the imaginary parts of the impedance
           M: differentiation matrix 
           data_used: part of the EIS spectrum used for regularization
           induct_used: treatment of the inductance part
       Output:
           kf score
    """
    
    lambda_value = exp(log_lambda)
    
    # non-negativity constraint on the DRT gamma
    # G@x<=0
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
        H_combined, c_combined = basics.quad_format_combined(A_re_train, A_im_train, Z_re_train, Z_im_train, M, lambda_value)
        args = [cvxopt.matrix(H_combined),cvxopt.matrix(c_combined), cvxopt.matrix(-bound_mat), cvxopt.matrix(lb)]
        
        # Solve the quadratic programming problems
        sol = cvxopt.solvers.qp(*args)
        if 'optimal' not in sol['status']:
            return None
        
        #solve for gamma
        gamma_ridge = np.array(sol['x']).flatten()
        
        # step 3: update of the kf scores    
        kf_cv += 1/Z_re_test.shape[0]*(norm(Z_re_test-A_re_test@gamma_ridge)**2 + norm(Z_im_test-A_im_test@gamma_ridge)**2)
    
    # kf score ; see section 1.2 in the supplementary information of [1]
    kf_cv_score = kf_cv/N_splits
    
    return kf_cv_score
compute_kf_cv.counter = 0


def compute_LC(log_lambda, A_re, A_im, Z_re, Z_im, M, data_used, induct_used):
    
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
           data_used: part of the EIS spectrum used for regularization
           induct_used: treatment of the inductance part
       Output:
           LC score
    """
    # 
    lambda_value = exp(log_lambda)
    
    A = np.vstack((A_re, A_im)) # matrix A with A_re and A_im; see (5) in [1]
    Z = np.hstack((Z_re, Z_im)) # stacked impedance
    
    A_in = A.T@A + lambda_value*M # see (13) in [1]
    
    # check if A_in is positive-definite
    if nPD.is_PD(A_in)==False:
        A_in = nPD.nearest_PD(A_in)
        
    # Cholesky transform to inverse A_in           
    L_in = cholesky(A_in) 
    
    # inverse of A_in based on the Cholesky transform of A_in
    inv_L_in = np.linalg.inv(L_in)
    inv_A_in = inv_L_in.T@inv_L_in # inverse of A_in
    A_LC = A@((inv_A_in.T@inv_A_in)@inv_A_in)@A.T
    
    # numerator eta_num of the first derivative of eta = log(||Z_exp - Ax||^2)
    eta_num = Z.T@A_LC@Z 

    A_in_d = A@A.T + lambda_value*np.eye(A.shape[0])
    
    # check if A_in_d is positive-definite  
    if nPD.is_PD(A_in_d)==False:
        A_in_d = nPD.nearest_PD(A_in_d)
        
    # Cholesky transform to inverse A_in_d    
    L_in_d = cholesky(A_in_d) 
    
    # inverse of A_in_d based on the Cholesky transform of A_in_d
    inv_L_in_d = np.linalg.inv(L_in_d)
    inv_A_in_d = inv_L_in_d.T@inv_L_in_d
    
    # denominator eta_denom of the first derivative of eta
    eta_denom = lambda_value*Z.T@(inv_A_in_d.T@inv_A_in_d)@Z
    
    # derivative of eta
    eta_prime = eta_num/eta_denom
    
    # numerator theta_num of the first derivative of theta = log(lambda*||Lx||^2)
    theta_num  = eta_num
    
    # denominator theta_denom of the first derivative of theta
    A_LC_d = A@(inv_A_in.T@inv_A_in)@A.T
    theta_denom = Z.T@A_LC_d@Z
    
    # derivative of theta 
    theta_prime = -(theta_num)/theta_denom
    
    # numerator LC_num of the LC score in (19) in [1]
    a_sq = (eta_num/(eta_denom*theta_denom))**2
    p = (Z.T@(inv_A_in_d.T@inv_A_in_d)@Z)*theta_denom
    m = (2*lambda_value*Z.T@((inv_A_in_d.T@inv_A_in_d)@inv_A_in_d)@Z)*theta_denom
    q = (2*lambda_value*Z.T@(inv_A_in_d.T@inv_A_in_d)@Z)*eta_num 
    LC_num = a_sq*(p+m-q)

    # denominator LC_denom of the LC score
    LC_denom = ((eta_prime)**2 + (theta_prime)**2)**(3/2)
    
    # LC score ; see (19) in [1]
    LC_score = LC_num/LC_denom
    
    return -LC_score 


