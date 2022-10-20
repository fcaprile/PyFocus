from typing import List
import numpy as np
from scipy.special import jv

def load_vortex_mask_functions(f, w0) -> List[callable]:
    gaussian=lambda theta:np.exp(-(np.sin(theta)*f/w0)**2)#incident gaussian beam's amplitude

    fun1=lambda theta, kr, kz: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)*(1 + np.cos(theta))*jv(1,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun2=lambda theta, kr, kz: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*jv(1,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun3=lambda theta, kr, kz: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)*(1 - np.cos(theta))*jv(3,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun4=lambda theta, kr, kz: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)**2*jv(0,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))
    fun5=lambda theta, kr, kz: gaussian(theta)*np.cos(theta)**0.5*np.sin(theta)**2*jv(2,kr*np.sin(theta))*np.exp(1j*kz*np.cos(theta))

    return [fun1, fun2, fun3, fun4, fun5]