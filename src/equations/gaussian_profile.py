import numpy as np

def gaussian_rho(w0):
    return lambda rho: np.exp(-(rho/w0)**2)

def gaussian_theta(f, w0):
    return lambda theta:np.exp(-(np.sin(theta)*f/w0)**2)