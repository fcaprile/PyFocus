from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from pydantic import BaseModel

from equations.complex_quadrature import complex_quadrature
from tqdm import tqdm

from equations.helpers import cart2pol
from src.dataclass.field_parameters import FieldParameters


class FreePropagationCalculator(ABC):
    
    @dataclass
    class ObjectiveFieldParameters:
        '''Parameters for the simulation of free propagation'''# TODO comentar aca que es cada parametro
        L: float # Distance between phase mask and objective lens (mm), only used if propagation=True
        R: float # Phase mask radius (mm), only used if propagation=True

        field_parameters: FieldParameters
        
        def transform_input_parameter_units(self):
            #transform to nanometers
            self.f*=10**6
            self.w0*=10**6
            
            #transform to radians:
            self.phip /= 180*np.pi
            self.field_parameters.transform_input_parameter_units()
    
    @abstractmethod
    def calculate(*args, **kwargs):
        raise NotImplementedError
    
    def calculate_amplitude_factors(parameters: ObjectiveFieldParameters): #TODO actualziar aprametros en llamada a funcion
        E1=np.sqrt(parameters.I0)*np.cos(parameters.field_parameters.polarization.gamma)/parameters.field_parameters.wavelength*np.pi
        E2=np.sqrt(parameters.I0)*np.sin(parameters.field_parameters.polarization.gamma)/parameters.field_parameters.wavelength*np.pi*np.exp(1j*parameters.field_parameters.polarization.beta)
        return E1, E2
