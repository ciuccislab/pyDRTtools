# -*- coding: utf-8 -*-
__authors__ = 'Francesco Ciucci, Ting Hei Wan, Baptiste Py, Adeleke Maradesa'

__date__ = '10th April 2024'

import numpy as np
from numpy import inf, pi
from numpy.random import randn
from numpy.linalg import cholesky
from numpy.matlib import repmat
from . import nearest_PD as nPD


def generate_tmg(F, g, M, mu_r, initial_X, cov=True, L=1):
    
        """
           This function returns samples from a truncated multivariate normal distribution.
           Reference: A. Pakman, L. Paninski, Exact hamiltonian Monte Carlo for truncated multivariate 
           Gaussians, J. Comput. Graph. Stat. 23 (2014) 518–542 (https://doi.org/10.48550/arXiv.1208.4118). 
           Inputs:
               F: m x d array (m is the number of constraints and d the dimension of the sample)
               g: m x 1 array 
               M: d x d array, which must be symmmetric and definite positive
               mu_r: d x 1 array 
               initial_X: d x 1 array, which must satisfy the constraint
               cov: condition to determine the covariance matrix, i.e., if cov == true, M is the covariance
                   matrix and mu_r the mean, while if cov == false M is the precision matrix and 
                   the log-density is -1/2*X'*M*X + r'*X
               L: number of samples desired
            Outputs:
               Xs: d x L array, each column is a sample being a sample from a d-dimensional Gaussian 
               with m constraints given by F*X+g >0 
        """
        
        # sanity check
        m = g.shape[0]
        if F.shape[0] != m:
            print("Error: constraint dimensions do not match")
            return

        # using covariance matrix
        if cov:
            mu = mu_r
            g = g + F@mu
            M = 0.5*(M + M.T) # symmetrize the matrix M
            if nPD.is_PD(M)==False:
                M = nPD.nearest_PD(M) 
            R = cholesky(M)
            R = R.T # change the lower matrix to upper matrix
            F = F@R.T
            initial_X = initial_X -mu
            initial_X = np.linalg.solve(R.T, initial_X)
            
        # using precision matrix
        else:
            r = mu_r
            M = 0.5*(M + M.T) # symmetrize the matrix M
            if nPD.is_PD(M)==False:
                M = nPD.nearest_PD(M)
            R = cholesky(M)
            R = R.T # change the lower matrix to upper matrix
            mu = np.linalg.solve(R, np.linalg.solve(R.T, r))
            g = g + F@mu
            F = np.linalg.solve(R, F)
            initial_X = initial_X - mu
            initial_X = R@initial_X

        d = initial_X.shape[0] # dimension of mean vector; each sample must be of this dimension
        bounce_count = 0
        nearzero = 1E-12
        
        # more for debugging purposes
        if (F@initial_X + g).any() < 0:
            print("Error: inconsistent initial condition")
            return

        # squared Euclidean norm of constraint matrix columns
        F2 = np.sum(np.square(F), axis=1)
        Ft = F.T
        
        last_X = initial_X
        Xs = np.zeros((d,L))
        Xs[:,0] = initial_X
        
        i=2
        
        # generate samples
        while i <=L:
            
            if i%1000 == 0:
                print('Current sample number',i,'/', L)
                
            stop = False
            j = -1
            # generate inital velocity from normal distribution
            V0 = randn(d)

            X = last_X
            T = pi/2
            tt = 0

            while True:
                a = np.real(V0)
                b = X

                fa = F@a
                fb = F@b

                U = np.sqrt(np.square(fa) + np.square(fb))
                # print(U.shape)

                # has to be arctan2 not arctan
                phi = np.arctan2(-fa, fb)

                # find the locations where the constraints were hit
                pn = np.array(np.abs(np.divide(g, U))<=1)
                
                if pn.any():
                    inds = np.where(pn)[0]
                    phn = phi[pn]
                    t1 = np.abs(-1.0*phn + np.arccos(np.divide(-1.0*g[pn], U[pn])))
                    
                    # if there was a previous reflection (j > -1)
                    # and there is a potential reflection at the sample plane
                    # make sure that a new reflection at j is not found because of numerical error
                    if j > -1:
                        if pn[j] == 1:
                            temp = np.cumsum(pn)
                            indj = temp[j]-1 # we changed this line
                            tt1 = t1[indj]
                            
                            if np.abs(tt1) < nearzero or np.abs(tt1 - pi) < nearzero:
                                # print(t1[indj])
                                t1[indj] = inf
                    
                    mt = np.min(t1)
                    m_ind = np.argmin(t1)
                    
                    # update j
                    j = inds[m_ind]
                     
                else:
                    mt = T
                
                # update travel time
                tt = tt + mt

                if tt >= T:
                    mt = mt- (tt - T)
                    stop = True

                # print(a)
                
                # update position and velocity
                X = a*np.sin(mt) + b*np.cos(mt)
                V = a*np.cos(mt) - b*np.sin(mt)

                if stop:
                    break
                
                # update new velocity
                qj = F[j,:]@V/F2[j]
                V0 = V - 2*qj*Ft[:,j]
                
                bounce_count += 1
        
            if (F@X +g).all() > 0:
                Xs[:,i-1] = X
                last_X = X
                i = i+1
    
            else:
                print('hmc reject')    
        
            # need to transform back to unwhitened frame
        if cov:
            Xs = R.T@Xs + repmat(mu.reshape(mu.shape[0],1),1,L)
        else:
            Xs =  np.linalg.solve(R, Xs) + repmat(mu.reshape(mu.shape[0],1),1,L)
        
        # convert back to array
        return Xs
    