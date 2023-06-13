from numpy import linalg as LA
import numpy as np

def soft_thresholding(x, gamma):
    if x > gamma:
        return x - gamma
    elif x < - gamma :
        return x + gamma
    else:
        return 0

def min_quadra(samples,label, rho, xi):
    """ Return the argmin of rho/2 *||samples w -label||^2 + || w - xi ||^2"""
    p = len(xi) 
    xi = np.reshape(xi, (1,p))

    if len(samples.shape) == 1:
        # compute the inverse of np.outer(samples, samples) + rho * np.eye(p)   
        inv = np.eye(p)/rho - np.outer(samples, samples)/(rho*rho + rho*samples@samples)  
        samples = np.reshape(samples, (1,p))
        label = np.reshape(label, (1,1))
    else:
        inv = np.linalg.inv(samples.T @ samples + rho * np.eye(p))

    num = np.reshape(label @ samples + rho * xi, (1,p)) 
    
    return np.reshape(inv @ num.T, (1,-1))

def lasso_obj(X, y, alpha, w):
    """Compute the value of the Lasso objective function"""
    return LA.norm(X @ w - y)**2/X.shape[0] + alpha * LA.norm(w, ord=1)


def myclip(v, L):
    """Returns v clipped to the parameter L """
    v_norm = LA.norm(v)
    if v_norm>L:
        return L*v/v_norm
    else:
        return v
    
def grad(samples, label, z):
    """Returns the gradient of the quadratic minimization"""
    return samples.T @ (samples @ z - label)