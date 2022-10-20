from abc import ABC, abstractclassmethod

import functools
from typing import Dict, List, Tuple
import numpy as np
from equations.complex_quadrature import complex_quadrature
from tqdm import tqdm

from dataclass.mask import MaskType
from equations.helpers import cart2pol


class DefaultMaskFocusFieldCalculator(ABC):
    
    @abstractclassmethod
    def execute(*args, **kwargs):
        raise NotImplementedError
    
    @abstractclassmethod
    def calculate_field(*args, **kwargs):
        raise NotImplementedError
    
    def _calculate_matrix_size(self, x_range: int, x_steps: int) -> Tuple[int, int]:
        return int(np.rint(x_range/x_steps/2-1)*2), int(np.rint(x_range/x_steps/2-1)*2)
    
    def _initialize_fields(self, x_size: int, y_size: int) -> List[list]:
        return [np.zeros((x_size, y_size),dtype=complex) for _ in range(6)]

    def mirror_on_z_axis(self, matrixes):
        N = matrixes[:,0,0]
        for i in range(N):
            matrix = matrixes[i]
            matrixes[i]=np.vstack((np.flipud(np.conj(matrix)),matrix[1:,:]))

    def calculate_steps(self, z_range, z_steps, x_range, r_steps):
        ztotalsteps=int(np.rint(z_range/z_steps/2))
        rtotalsteps=int(np.rint(x_range/r_steps/2**0.5)) #the actual field of view of the X axis in the XZ plane will be x_range*2**0.5
        return ztotalsteps, rtotalsteps
    
    def integrate(self, matrix_ammount, functions_to_integrate, wavelength, alpha, z_range, x_range, ztotalsteps, rtotalsteps, description):
        matrixes = [np.zeros((ztotalsteps, rtotalsteps),dtype=complex) for _ in range(matrix_ammount)]
        
        for n_z in tqdm(range(ztotalsteps),desc=description):
            for n_r in range(rtotalsteps):
                kz=n_z*2*np.pi/wavelength/ztotalsteps*z_range/2 
                kr=n_r*2*np.pi/wavelength/rtotalsteps*x_range/2*2**0.5
                
                for i in range(matrix_ammount):
                    (matrixes[i])[n_z,n_r] = complex_quadrature(functools.partial(functions_to_integrate[i], kz, kr),0,alpha)[0]
    
    def calculate_factors(I0, gamma, beta, wavelength, f):
        E1=np.sqrt(I0)*np.cos(gamma)/wavelength*np.pi*f
        E2=np.sqrt(I0)*np.sin(gamma)/wavelength*np.pi*f*np.exp(1j*beta)
        return E1, E2
