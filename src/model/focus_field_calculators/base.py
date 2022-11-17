from abc import ABC, abstractclassmethod
from pydantic import BaseModel, StrictFloat

import functools
from typing import Dict, List, Tuple
import numpy as np
from equations.complex_quadrature import complex_quadrature
from tqdm import tqdm

from custom_dataclasses.field_parameters import FieldParameters
from dataclasses import dataclass

class FocusFieldCalculator(ABC):
    
    @dataclass
    class FieldAtFocus:
        Ex_XZ: List(list) # Ex component at the XZ plane
        Ey_XZ: List(list) # Ey component at the XZ plane
        Ez_XZ: List(list) # Ez component at the XZ plane
        Ex_XY: List(list) # Ex component at the XY plane
        Ey_XY: List(list) # Ey component at the XY plane
        Ez_XY: List(list) # Ez component at the XY plane
        
        def calculate_intensity(self):
            self.Intensity_XZ = np.abs(self.Ex_XZ)**2+np.abs(self.Ey_XZ)**2+np.abs(self.Ez_XZ)**2
            self.Intensity_XY = np.abs(self.Ex_XY)**2+np.abs(self.Ey_XY)**2+np.abs(self.Ez_XY)**2
        
    
    class FocusFieldParameters(BaseModel):
        '''Parameters for the simulation of the field near the focus'''# TODO comentar aca que es cada parametro
        NA: StrictFloat # Numerical aperture
        n: StrictFloat # Refraction index for the medium of the optical system.
        h: StrictFloat # Radius of aperture of the objective lens
        f: StrictFloat # Focal distance
        z: StrictFloat # Axial position for the XY plane (nm)
        x_steps: StrictFloat # Resolution in the X or Y coordinate for the focused field (nm)
        z_steps: StrictFloat # Resolution in the axial coordinate (Z) for the focused field (nm)
        x_range: StrictFloat # Field of view in the X or Y coordinate in which the focused field is calculated (nm)
        z_range: StrictFloat # Field of view in the axial coordinate (z) in which the focused field is calculated (nm)
        phip: StrictFloat # Angle of rotation along the z axis when calculating the field
        
        field_parameters: FieldParameters
        
        def transform_input_parameter_units(self):
            #transform to nanometers
            self.f*=10**6
            self.w0*=10**6 # TODO ver unidades campo gaussiano incidente
            
            #transform to radians:
            self.phip /= 180*np.pi
            self.field_parameters.transform_input_parameter_units()

    @abstractclassmethod
    def calculate(*args, **kwargs) -> FieldAtFocus:
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
    
    def calculate_amplitude_factors(parameters: FocusFieldParameters): #TODO actualziar aprametros en llamada a funcion
        E1=np.sqrt(parameters.I0)*np.cos(parameters.field_parameters.polarization.gamma)/parameters.field_parameters.wavelength*np.pi*parameters.f
        E2=np.sqrt(parameters.I0)*np.sin(parameters.field_parameters.polarization.gamma)/parameters.field_parameters.wavelength*np.pi*parameters.f*np.exp(1j*parameters.field_parameters.polarization.beta)
        return E1, E2
