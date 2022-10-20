from abc import ABC, abstractclassmethod
from src.equations.helpers.plot.plot_objective_field import plot_objective_field

import functools
from typing import Dict, List, Tuple
import numpy as np
from equations.complex_quadrature import complex_quadrature
from tqdm import tqdm

from dataclass.mask import MaskType
from equations.helpers import cart2pol


class DefaultMaskFreePropagationCalculator(ABC):
    
    def __init__(self, plot_function: callable = None) -> None:
        self.plot_function = plot_function if plot_function else plot_objective_field
    
    @abstractclassmethod
    def execute(*args, **kwargs):
        raise NotImplementedError
        
    def calculate_factors(I0, gamma, beta, wavelength):
        E1=np.sqrt(I0)*np.cos(gamma)/wavelength*np.pi
        E2=np.sqrt(I0)*np.sin(gamma)/wavelength*np.pi*np.exp(1j*beta)
        return E1, E2
