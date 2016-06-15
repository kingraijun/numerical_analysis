import numpy as np
from numpy import linalg as npl
import pandas as pd
from copy import deepcopy

import matrix_test as mt

"""
        The Power method is an iterative technique used to determine the dominant eigenvalue of a matrix -- that is, the eigenvalue with the largest magnitude. By modifying the method slightly, it can also used to determine other eigenvalues. One useful feature of the Power method is that it produces not only an eigenvalue, but also an associated eigenvector. In fact, the Power method is often applied to find an eigenvector for an eigenvalue that is determined by some other means.
"""

def _append_val(x, mu, mu2):
    if isinstance(mu, (int,float)):
        mu = np.around(mu,decimals=6)
    if isinstance(mu2, (int,float)):
        mu2 = np.around(mu2,decimals=6)
    return pd.DataFrame({'Eigenvector':[np.around(x,decimals=6)], 
                         'Eigenvalue':mu,
                         'Eigenvalue_Aitkens':mu2},
                        columns=['Eigenvector','Eigenvalue', 'Eigenvalue_Aitkens'])

def powermethod(A, x, TOL=1e-05, maxIter=20):
    """
        To approximate the dominant eigenvalue and an associated eigenvector of the n by n matrix A given a nonzero vector x.
    
        Parameters
        ----------
        A : array-like
            The input matrix.
        x : array-like
            The initial vector.
        TOL : float, 1e-05
            The tolerance.
        maxIter : int, 20
            The maximum iteration before the algorithm terminates.
            
        Returns
        -------
        mu : float
            The approximate eigenvalue.
        x_mu : array_like
            The approximate eigenvector (with norm of x_mu to be 1).
    """
    
    assert mt.isSquare(A), 'Input matrix is not a square matrix'         
    
    y = np.zeros_like(x)
    mu_0 = mu_1 = 0
    
    res_table = _append_val(x,'','')
    
    p = min([i if abs(j) == npl.norm(x, np.inf) else np.inf for i,j in enumerate(x)])
    x_ = x/x[p]
        
    for k in range(maxIter):
        y = np.dot(A, x_)
        mu = y[p]
        mu_hat = mu_0 - ((mu_1-mu_0)**2/(mu-2*mu_1+mu_0))
        
        p = min([i if abs(j) == npl.norm(y, np.inf) else np.inf for i,j in enumerate(y)])

        if y[p] == 0:
            print('Input matrix has the eigenvalue 0. Select a new vector x and restart.')
            return res_table
        
        ERR = npl.norm(x_ - y/y[p], np.inf)
        x_ = y/y[p]
        
        res_table = res_table.append(_append_val(x_,mu,mu_hat), ignore_index=True)   
        
        if ERR < TOL and k>=3:
            print('The procedure was successful. With TOLERANCE set at' \
                  + ' %f, algorithm terminated after %d iterations.'%(TOL, k+1))
            return res_table
        
        mu_0, mu_1 = mu_1, mu
    
    print('The maximum number of iterations (maxIter=%d) reached.'%(maxIter))
    
    return res_table
    
def symmetric_powermethod(A, x, TOL=1e-05, maxIter=20):
    """
        To approximate the dominant eigenvalue and an associated eigenvector of the n by n symmetric matrix A given a nonzero vector x.
    
        Parameters
        ----------
        A : array-like
            The input matrix.
        x : array-like
            The initial vector.
        TOL : float, 1e-05
            The tolerance.
        maxIter : int, 20
            The maximum iteration before the algorithm terminates.
            
        Returns
        -------
        mu : float
            The approximate eigenvalue.
        x_mu : array_like
            The approximate eigenvector (with norm of x_mu to be 1).
    """

    assert mt.isSymmetric(A), 'Input matrix is not symmetric.'    
    
    y = np.zeros_like(x)
    mu_0 = mu_1 = 0
    
    res_table = _append_val(x,'','')
    
    x_ = x/npl.norm(x)
    
    for k in range(maxIter):
        y = np.dot(A, x_)
        mu = np.dot(x_.T,y)
        mu_hat = mu_0 - ((mu_1-mu_0)**2/(mu-2*mu_1+mu_0))

        if npl.norm(y) == 0:
            print('Input matrix has the eigenvalue 0. Select a new vector x and restart.')
            return res_table
        
        ERR = npl.norm(x - y/npl.norm(y))
        x_ = y/npl.norm(y)
        
        res_table = res_table.append(_append_val(x_,mu,mu_hat), ignore_index=True) 
        
        if ERR < TOL and k>=3:
            print('The procedure was successful. With TOLERANCE set at' \
                  + ' %f, algorithm terminated after %d iterations.'%(TOL, k+1))
            return res_table
        
        mu_0, mu_1 = mu_1, mu
     
    print('The maximum number of iterations (maxIter=%d) reached.'%(maxIter))
    
    return res_table