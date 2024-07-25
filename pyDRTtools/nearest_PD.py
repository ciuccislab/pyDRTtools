
__authors__ = 'Francesco Ciucci, Adeleke Maradesa, Baptiste Py'

__date__ = '10th April 2024'


import numpy as np


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
        min_eig = min(0, np.min(np.real(np.linalg.eigvalsh(A_symm))))
        A_symm += I * (-min_eig * k**2 + eps)
        k += 1

    return A_symm