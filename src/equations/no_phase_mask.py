import numpy as np
from scipy.special import jv

from src.equations.gaussian_profile import gaussian_theta

def load_no_mask_functions(f, w0) -> list[callable]:
    gaussian=gaussian_theta(f, w0)

    fun1=lambda theta, kr, kz: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)*(1+np.cos(theta))*jv(0,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun2=lambda theta, kr, kz: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)**2*jv(1,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun3=lambda theta, kr, kz: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)*(1-np.cos(theta))*jv(2,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))

    return [fun1, fun2, fun3]
