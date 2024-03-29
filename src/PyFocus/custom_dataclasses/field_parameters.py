from dataclasses import dataclass
import numpy as np

@dataclass
class PolarizationParameters:
    ''' Parameters that define the polarization of the inciding laser beam '''
    gamma: float # Parameter that determines the polarization, arctan(ey/ex) (gamma=45, beta=90 gives right circular polarization and a donut shape)
    beta: float # Parameter that determines the polarization, phase difference between ex and ey (gamma=45, beta=90 gives right circular polarization and a donut shape)

@dataclass
class FieldParameters:
    ''' Parameters that define the inciding laser beam '''
    w0: float # Radius of the incident gaussian beam
    wavelength: float # Wavelength in vacuum
    I_0: float # Incident field intensity (mW/cm^2)
    polarization: PolarizationParameters
    
    def transform_input_parameter_units(self):
        '''transform input unis from degrees to radians and from milimeters to nanometers'''
        self.w0 *= 10**6
        
        self.polarization.beta *= np.pi/180
        self.polarization.gamma *= np.pi/180

@dataclass
class InterfaseParameters:
    ''' Parameters that define an interface located near the focus '''
    ds: list # Thickness of each interface in the multilayer system (nm), must be given as a numpy array with the first and last values a np.inf. Only used if interface=True
    z_int: float # Axial position of the interphase. Only used if interface=True
