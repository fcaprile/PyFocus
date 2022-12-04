from abc import ABC, abstractclassmethod
from copy import deepcopy
from pydantic import BaseModel, StrictFloat, StrictInt
from custom_typing import Matrix

import functools
from typing import Any, Dict, Tuple, Union
import numpy as np
from equations.complex_quadrature import complex_quadrature
from tqdm import tqdm

from custom_dataclasses.field_parameters import FieldParameters
from dataclasses import dataclass

class FocusFieldCalculator(ABC):
    
    @dataclass
    class FieldAtFocus:
        Ex_XZ: Matrix # Ex component at the XZ plane
        Ey_XZ: Matrix # Ey component at the XZ plane
        Ez_XZ: Matrix # Ez component at the XZ plane
        Ex_XY: Matrix # Ex component at the XY plane
        Ey_XY: Matrix # Ey component at the XY plane
        Ez_XY: Matrix # Ez component at the XY plane
        
        def calculate_intensity(self):
            self.Intensity_XZ = np.abs(self.Ex_XZ)**2+np.abs(self.Ey_XZ)**2+np.abs(self.Ez_XZ)**2
            self.Intensity_XY = np.abs(self.Ex_XY)**2+np.abs(self.Ey_XY)**2+np.abs(self.Ez_XY)**2
        
        def calculate_intensity_along_x(self):
            self.Intensity_along_x=self.Intensity_XY[int(np.shape(self.Intensity_XY)[0]/2),:]
    
    class FocusFieldParameters(BaseModel):
        '''Parameters for the simulation of the field near the focus'''# TODO comentar aca que es cada parametro
        NA: Union[StrictFloat, StrictInt] # Numerical aperture
        n: Union[StrictFloat, StrictInt] # Refraction index for the medium of the optical system.
        h: Union[StrictFloat, StrictInt] # Radius of aperture of the objective lens
        f: Union[StrictFloat, StrictInt] = 0 # Focal distance
        
        x_steps: Union[StrictFloat, StrictInt] # Resolution in the X or Y coordinate for the focused field (nm)
        z_steps: Union[StrictFloat, StrictInt] # Resolution in the axial coordinate (Z) for the focused field (nm)
        x_range: Union[StrictFloat, StrictInt] # Field of view in the X or Y coordinate in which the focused field is calculated (nm)
        z_range: Union[StrictFloat, StrictInt] # Field of view in the axial coordinate (z) in which the focused field is calculated (nm)
        
        z: Union[StrictFloat, StrictInt] # Axial position for the XY plane (nm)
        phip: Union[StrictFloat, StrictInt] # Angle of rotation along the z axis when calculating the field
        
        field_parameters: FieldParameters
        
        @property
        def alpha(self) -> float:
            return np.arcsin(self.NA / self.n)
        
        @property
        def ztotalsteps(self) -> int:
            return int(np.rint(self.z_range/self.z_steps/2))
        
        @property
        def rtotalsteps(self) -> int:
            return int(np.rint(self.x_range/self.x_steps/2**0.5)) #the actual field of view of the X axis in the XZ plane will be x_range*2**0.5

        
        def transform_input_parameter_units(self):
            '''transform to radians and from milimeters to nanometers'''
            self.field_parameters.transform_input_parameter_units()
            
            self.f = self.h * self.n/ self.NA *10**6
            
            #transform to radians:
            self.phip /= 180*np.pi

    @abstractclassmethod
    def calculate(focus_field_parameters: FocusFieldParameters) -> FieldAtFocus:
        raise NotImplementedError
    
    @abstractclassmethod
    def _calculate_field(*args, **kwargs):
        raise NotImplementedError
    
    def _calculate_matrix_size(self, x_range: int, x_steps: int) -> Tuple[int, int]:
        return int(np.rint(x_range/x_steps/2-1)*2), int(np.rint(x_range/x_steps/2-1)*2)
    
    def _initialize_fields(self, x_size: int, y_size: int) -> list[list]:
        return [np.zeros((x_size, y_size),dtype=complex) for _ in range(6)]

    def mirror_on_z_axis(self, matrixes):
        output = []
        for matrix in deepcopy(matrixes):
            output.append(np.vstack((np.flipud(np.conj(matrix)),matrix[1:,:])))
        return output
    
    def integrate(self, matrix_ammount, functions_to_integrate, focus_field_parameters: FocusFieldParameters, description):
        
        matrixes = [np.zeros((focus_field_parameters.ztotalsteps, focus_field_parameters.rtotalsteps),dtype=complex) for _ in range(matrix_ammount)]
        
        print(f'{focus_field_parameters.ztotalsteps=}')
        print(f'{focus_field_parameters=}')
        print(f'{focus_field_parameters.alpha=}')
        for n_z in tqdm(range(focus_field_parameters.ztotalsteps),desc=description): #TODO ir ploteando los valores que toma n_z
            for n_r in range(focus_field_parameters.rtotalsteps):
                kz=n_z*2*np.pi/focus_field_parameters.field_parameters.wavelength/focus_field_parameters.ztotalsteps*focus_field_parameters.z_range/2 
                kr=n_r*2*np.pi/focus_field_parameters.field_parameters.wavelength/focus_field_parameters.rtotalsteps*focus_field_parameters.x_range/2*2**0.5
                
                for i, matrix in enumerate(matrixes):
                    result = complex_quadrature(functools.partial(functions_to_integrate[i], kz=kz, kr=kr),0,focus_field_parameters.alpha)[0]
                    #print(f'{i=},{result=}')
                    (matrixes[i])[n_z,n_r] = result
        
        return matrixes
    
    def _calculate_amplitude_factors(self, parameters: FocusFieldParameters): #TODO actualziar parametros en llamada a funcion
        E1=np.sqrt(parameters.field_parameters.I_0)*np.cos(parameters.field_parameters.polarization.gamma)/parameters.field_parameters.wavelength*np.pi*parameters.f
        E2=np.sqrt(parameters.field_parameters.I_0)*np.sin(parameters.field_parameters.polarization.gamma)/parameters.field_parameters.wavelength*np.pi*parameters.f*np.exp(1j*parameters.field_parameters.polarization.beta)
        return E1, E2
