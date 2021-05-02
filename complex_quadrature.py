import numpy as np
from scipy.integrate import quad

def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b, limit=2000)
    imag_integral = quad(imag_func, a, b, limit=2000)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])
