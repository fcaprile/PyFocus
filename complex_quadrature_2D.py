import numpy as np
from scipy.integrate import dblquad


def dbl_complex_quadrature(func, a, b,c,d, **kwargs):
    def real_func(x,y):
        return np.real(func(x,y))
    def imag_func(x,y):
        return np.imag(func(x,y))
    real_integral = dblquad(real_func, a, b,c,d, **kwargs)
    imag_integral = dblquad(imag_func, a, b,c,d, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])
