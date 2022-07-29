import numpy as np
from numpy import exp
from math import pi, log, sqrt
from scipy import integrate
from scipy.optimize import fsolve
from scipy.linalg import toeplitz, hankel
import cvxpy as cp
import cvxopt

"""
this file store all the functions that are shared by all the three DRT method, i.e., simple, Bayesian, and BHT
#####need to ensure that every function read the same as the pyDRTtools#####

"""

def g_i(freq_n, tau_m, epsilon, rbf_type):
    """
        this function generate the elements of A_re    
    """
    alpha = 2*pi*freq_n*tau_m  
    
    rbf_switch = {
                'Gaussian': lambda x: exp(-(epsilon*x)**2),
                'C0 Matern': lambda x: exp(-abs(epsilon*x)),
                'C2 Matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                'C4 Matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                'C6 matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                'Inverse Quadratic': lambda x: 1/(1+(epsilon*x)**2),
                'Inverse Quadric': lambda x: 1/sqrt(1+(epsilon*x)**2),
                'Cauchy': lambda x: 1/(1+abs(epsilon*x))
                }
    
    rbf = rbf_switch.get(rbf_type)
    integrand_g_i = lambda x: 1./(1.+(alpha**2)*exp(2.*x))*rbf(x)
    out_val = integrate.quad(integrand_g_i, -50, 50, epsabs=1E-9, epsrel=1E-9)
    
    return out_val[0]


def g_ii(freq_n, tau_m, epsilon, rbf_type):
    """
       this function generate the elements of A_im 
    """ 
    alpha = 2*pi*freq_n*tau_m  
    
    rbf_switch = {
                'Gaussian': lambda x: exp(-(epsilon*x)**2),
                'C0 Matern': lambda x: exp(-abs(epsilon*x)),
                'C2 Matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                'C4 Matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                'C6 matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                'Inverse Quadratic': lambda x: 1/(1+(epsilon*x)**2),
                'Inverse Quadric': lambda x: 1/sqrt(1+(epsilon*x)**2),
                'Cauchy': lambda x: 1/(1+abs(epsilon*x))
                }
    
    rbf = rbf_switch.get(rbf_type)
    integrand_g_ii = lambda x: alpha/(1./exp(x)+(alpha**2)*exp(x))*rbf(x)    
    out_val = integrate.quad(integrand_g_ii, -50, 50, epsabs=1E-9, epsrel=1E-9)
    
    return out_val[0]


def compute_epsilon(freq, coeff, rbf_type, shape_control): 
    """
        this function is used to compute epsilon, i.e., the shape factor of
        the rbf used for discretization. user can directly set the shape factor
        by selecting 'shape' for the shape_control. alternatively, 
        when 'FWHM_coeff' is selected, the shape factor is such that 
        the full width half maximum (FWHM) of the rbf equals to the average 
        relaxation time spacing in log space over coeff, i.e., FWHM = delta(ln tau)/coeff
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
    
    if shape_control == 'FWHM Coefficient':
        #equivalent as the 'FWHM Coefficient' option in matlab code
        FWHM_coeff = 2*fsolve(rbf,1)
        delta = np.mean(np.diff(np.log(1/freq.reshape(N_freq))))
        epsilon = coeff*FWHM_coeff/delta
        
    else:
        #equivalent as the 'Shape Factor' option in matlab code
        epsilon = coeff
    
    return epsilon[0]
    

def inner_prod_rbf_1(freq_n, freq_m, epsilon, rbf_type):
    """  
        this function output the inner product of the first derivative of the
        rbf with respect to log(1/freq_n) and log(1/freq_m)
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
#        out_IP = integral(@(y) sqr_drbf_dy(y),-Inf,Inf);    
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
        this function output the inner product of the second derivative of the
        rbf with respect to log(1/freq_n) and log(1/freq_m)
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
        # MATLAB code:     out_IP = integral(@(y) sqr_drbf_dy(y),-Inf,Inf);    
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


def gamma_to_x(gamma_vec, tau_vec, epsilon, rbf_type): ## double check this to see if the function is correct
    """  
        this function map the gamma_vec back to the x vector
        for piecewise linear, x = gamma
    """  
    if rbf_type == 'Piecewise Linear':
        x_vec = gamma_vec
        
    else:
        rbf_switch = {
                'Gaussian': lambda x: exp(-(epsilon*x)**2),
                'C0 Matern': lambda x: exp(-abs(epsilon*x)),
                'C2 Matern': lambda x: exp(-abs(epsilon*x))*(1+abs(epsilon*x)),
                'C4 Matern': lambda x: 1/3*exp(-abs(epsilon*x))*(3+3*abs(epsilon*x)+abs(epsilon*x)**2),
                'C6 matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
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


def x_to_gamma(x_vec, tau_map_vec, tau_vec, epsilon, rbf_type): ## double check this to see if the function is correct
    """  
        this function map the gamma_vec back to the x vector
        for Piecewise Linear, x = gamma
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
                    'C6 matern': lambda x: 1/15*exp(-abs(epsilon*x))*(15+15*abs(epsilon*x)+6*abs(epsilon*x)**2+abs(epsilon*x)**3),
                    'Inverse Quadratic': lambda x: 1/(1+(epsilon*x)**2),
                    'Inverse Quadric': lambda x: 1/sqrt(1+(epsilon*x)**2),
                    'Cauchy': lambda x: 1/(1+abs(epsilon*x))
                    }
        
        rbf = rbf_switch.get(rbf_type)
        
        N_taus = tau_vec.size
        N_tau_map = tau_map_vec.size
        gamma_vec = np.zeros([N_tau_map, 1])
#        rbf_vec = np.zeros([N_taus,1])
        B = np.zeros([N_tau_map, N_taus])
        
        for p in range(0, N_tau_map):
            for q in range(0, N_taus):
                delta_log_tau = log(tau_map_vec[p])-log(tau_vec[q])
                B[p,q] = rbf(delta_log_tau)              
#        B = 0.5*(B+B.T)              
        gamma_vec = B@x_vec
        out_tau_vec = tau_map_vec 
        
    return out_tau_vec,gamma_vec


def assemble_A_re(freq_vec, tau_vec, epsilon, rbf_type):
    """
        this function assemble the A_re matrix
    """    
#   compute number of frequency, tau and omega
    omega_vec = 2.*pi*freq_vec
    N_freqs = freq_vec.size
    N_taus = tau_vec.size

#   define the A_re output matrix
    out_A_re = np.zeros((N_freqs, N_taus))
    
#   check if the frequencies are sufficiently log spaced
    std_diff_freq = np.std(np.diff(np.log(1/freq_vec)))
    mean_diff_freq = np.mean(np.diff(np.log(1/freq_vec)))

#   check if the frequencies are sufficiently log spaced and that N_freqs = N_taus
    toeplitz_trick = std_diff_freq/mean_diff_freq<0.01 and N_freqs == N_taus 

    if toeplitz_trick and rbf_type != 'Piecewise Linear':
        # use toeplitz trick
        R = np.zeros(N_taus)
        C = np.zeros(N_freqs)
        
        for p in range(0, N_freqs):
            
            C[p] = g_i(freq_vec[p], tau_vec[0], epsilon, rbf_type)
        
        for q in range(0, N_taus):
            
            R[q] = g_i(freq_vec[0], tau_vec[q], epsilon, rbf_type)        
                        
        out_A_re= toeplitz(C,R) 

    else:
        # use brute force
        for p in range(0, N_freqs):
            for q in range(0, N_taus):
            
                if rbf_type == 'Piecewise Linear':                
                    if q == 0:
                        out_A_re[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                    elif q == N_taus-1:
                        out_A_re[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                    else:
                        out_A_re[p, q] = 0.5/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])                    
                
                else:
                    out_A_re[p, q]= g_i(freq_vec[p], tau_vec[q], epsilon, rbf_type)

    return out_A_re


def assemble_A_im(freq_vec, tau_vec, epsilon, rbf_type):
    """
        This function assemble the A_im matrix
    """        
#   compute number of frequency, tau and omega
    omega_vec = 2.*pi*freq_vec
    N_freqs = freq_vec.size
    N_taus = tau_vec.size

#   define the A_re output matrix
    out_A_im = np.zeros((N_freqs, N_taus))
    
#   check if the frequencies are sufficiently log spaced
    std_diff_freq = np.std(np.diff(np.log(1/freq_vec)))
    mean_diff_freq = np.mean(np.diff(np.log(1/freq_vec)))

#   check if the frequencies are sufficiently log spaced and that N_freqs = N_taus
    toeplitz_trick = std_diff_freq/mean_diff_freq<0.01 and N_freqs == N_taus 
    
    if toeplitz_trick and rbf_type != 'Piecewise Linear':
        # use toeplitz trick
        R = np.zeros(N_taus)
        C = np.zeros(N_freqs)
        
        for p in range(0, N_freqs):
            
            C[p] = - g_ii(freq_vec[p], tau_vec[0], epsilon, rbf_type)
        
        for q in range(0, N_taus):
            
            R[q] = - g_ii(freq_vec[0], tau_vec[q], epsilon, rbf_type)        
                        
        out_A_im = toeplitz(C,R) 

    else:
        # use brute force
        for p in range(0, N_freqs):
            for q in range(0, N_taus):
            
                if rbf_type == 'Piecewise Linear':                
                    if q == 0:
                        out_A_im[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q])
                    elif q == N_taus-1:
                        out_A_im[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q]/tau_vec[q-1])
                    else:
                        out_A_im[p, q] = -0.5*(omega_vec[p]*tau_vec[q])/(1+(omega_vec[p]*tau_vec[q])**2)*log(tau_vec[q+1]/tau_vec[q-1])                    
                
                else:
                    out_A_im[p, q]= - g_ii(freq_vec[p], tau_vec[q], epsilon, rbf_type)
 
    
    return out_A_im


def assemble_M_1(tau_vec, epsilon, rbf_type):
    """
        this function assembles the M matrix which contains the 
        the inner products of 1st-derivative of the discretization rbfs
        size of M matrix depends on the number of collocation points, i.e. tau vector
    """
    freq_vec = 1/tau_vec   
    # first get number of collocation points
    N_taus = tau_vec.size
    N_freq = freq_vec.size
    # define the M output matrix
    out_M = np.zeros([N_taus, N_taus])
    
    #check if the collocation points are sufficiently log spaced
    std_diff_freq = np.std(np.diff(np.log(tau_vec)));
    mean_diff_freq = np.mean(np.diff(np.log(tau_vec)));
    
    #If they are we apply the toeplitz trick   
    toeplitz_trick = std_diff_freq/mean_diff_freq<0.01
    
    if toeplitz_trick and rbf_type != 'Piecewise Linear':
        #Apply the toeplitz trick to compute the M matrix 
        R = np.zeros(N_taus)
        C = np.zeros(N_taus)
        
        for n in range(0,N_taus):
            C[n] = inner_prod_rbf_1(freq_vec[0], freq_vec[n], epsilon, rbf_type)# may be use tau instead of freq
            
        for m in range(0,N_taus):
            R[m] = inner_prod_rbf_1(freq_vec[m], freq_vec[0], epsilon, rbf_type)    
        
        out_M = toeplitz(C,R) 
         
    elif rbf_type == 'Piecewise Linear':
        #If piecewise linear discretization
        out_L_temp = np.zeros([N_freq-1, N_freq])
        
        for iter_freq_n in range(0,N_freq-1):
            delta_loc = log((1/freq_vec[iter_freq_n+1])/(1/freq_vec[iter_freq_n]))
            out_L_temp[iter_freq_n,iter_freq_n] = -1/delta_loc
            out_L_temp[iter_freq_n,iter_freq_n+1] = 1/delta_loc

        out_M = out_L_temp.T@out_L_temp
    
    else:
        #compute rbf with brute force
        for n in range(0, N_taus):
            for m in range(0, N_taus):            
                out_M[n,m] = inner_prod_rbf_1(freq_vec[n], freq_vec[m], epsilon, rbf_type)
        
    return out_M


def assemble_M_2(tau_vec, epsilon, rbf_type):
    """
        this function assembles the M matrix which contains the 
        the inner products of 2nd-derivative of the discretization rbfs
        size of M matrix depends on the number of collocation points, i.e. tau vector
    """ 
    freq_vec = 1/tau_vec            
    # first get number of collocation points
    N_freqs = freq_vec.size
    N_taus = tau_vec.size
    
    # define the M output matrix
    out_M = np.zeros([N_taus, N_taus])
    
    #check if the collocation points are sufficiently log spaced
    std_diff_freq = np.std(np.diff(np.log(tau_vec)));
    mean_diff_freq = np.mean(np.diff(np.log(tau_vec)));
    
    #If they are we apply the toeplitz trick   
    toeplitz_trick = std_diff_freq/mean_diff_freq<0.01
    
    if toeplitz_trick and rbf_type != 'Piecewise Linear':
        #Apply the toeplitz trick to compute the M matrix 
        R = np.zeros(N_taus)
        C = np.zeros(N_taus)
        
        for n in range(0,N_taus):
            C[n] = inner_prod_rbf_2(freq_vec[0], freq_vec[n], epsilon, rbf_type)# later, we shall use tau instead of freq
            
        for m in range(0,N_taus):
            R[m] = inner_prod_rbf_2(freq_vec[m], freq_vec[0], epsilon, rbf_type)# later, we shall use tau instead of freq
        
        out_M = toeplitz(C,R) 
         
    elif rbf_type == 'Piecewise Linear':
        #Piecewise linear discretization
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
    
    else:
        #compute rbf with brute force
        for n in range(0, N_taus):
            for m in range(0, N_taus):            
                out_M[n,m] = inner_prod_rbf_2(freq_vec[n], freq_vec[m], epsilon, rbf_type)
        
    
    return out_M


def quad_format(A,b,M,lambda_value):
    """
        this function reformats the DRT regression 
        as a quadratic program - this uses either re or im
    """
    H = 2*(A.T@A+lambda_value*M)
    H = (H.T+H)/2
    c = -2*b.T@A
    
    return H,c


def quad_format_combined(A_re,A_im,b_re,b_im,M,lambda_value): 
    """
        this function reformats the DRT regression 
        as a quadratic program - this uses both re and im
    """
    H = 2*((A_re.T@A_re+A_im.T@A_im)+lambda_value*M)
    H = (H.T+H)/2
    c = -2*(b_im.T@A_im+b_re.T@A_re)

    return H,c


def cvxpy_solve_qp(H, c):
    """
        this function conducts the quadratic programming with cvxpy and
        output the optimum in numpy array format
    """
    N_out = c.shape[0]
    x = cp.Variable(shape = N_out, value = np.ones(N_out))
    h = np.zeros(N_out)

    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, H) + c@x), [x >= h])
    prob.solve(verbose = True, eps_abs = 1E-10, eps_rel = 1E-10, sigma = 1.00e-08, 
               max_iter = 200000, eps_prim_inf = 1E-5, eps_dual_inf = 1E-5)

    gamma = x.value
    
    return gamma


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
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    
    return np.array(sol['x']).reshape((P.shape[1],))




def pretty_plot(width=8, height=None, plt=None, dpi=None,
                color_cycle=("qualitative", "Set1_9")):
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

