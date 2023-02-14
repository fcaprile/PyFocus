import os
import sys
sys.path.append(os.path.abspath("./src"))

from model.focus_field_calculators.base import FocusFieldCalculator
from model.free_propagation_calculators.base import FreePropagationCalculator
from custom_dataclasses.field_parameters import FieldParameters, PolarizationParameters
from custom_dataclasses.mask import MaskType
from custom_dataclasses.interface_parameters import InterfaceParameters
import numpy as np
from model.main_calculation_handler import MainCalculationHandler

class PyFocusSimulator:
    
    def __init__(self,NA, n, wavelength, Nxy, Nz, dr, dz) -> None:
        self.NA = NA
        self.n = n
        self.wavelength = wavelength
        self.Nxy = Nxy
        self.Nz = Nz
        self.dr = dr
        self.dz = dz
        
        self.add_Zernike_aberration = None
        
        self._transform_units()
    
    def _transform_units(self):
        ''' Performs a passage from um to nm and calculates the FOV'''
        self.wavelength*=1000 
        self.dr *= 1000
        self.dz *= 1000
        
        self.radial_FOV = self.dr*self.Nxy
        self.axial_FOV = self.dz*self.Nz
    
    def generate_pupil(self):
        '''Generates the field inciding on the objective lens'''
        pass
    
    def generate_3D_PSF(self):
        fields = ...
        self.PSF3D = fields
    
    def add_slab_scalar(self):
        pass