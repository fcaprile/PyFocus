from abc import ABC, abstractclassmethod
from copy import deepcopy
from pydantic import BaseModel, StrictFloat, StrictInt
from ...custom_typing import Matrix, Matrix3D

import functools
from typing import Any, Dict, Optional, Tuple, Union, List
import numpy as np
from ...equations.complex_quadrature import complex_quadrature
from tqdm import tqdm

from ...custom_dataclasses.interface_parameters import InterfaceParameters
from ...custom_dataclasses.field_parameters import FieldParameters
from ...custom_dataclasses.custom_mask import CustomMaskParameters
from dataclasses import dataclass

class FocusFieldCalculator(ABC):
    
    @dataclass
    class FieldAtFocus:
        Ex_XZ: Optional[Matrix] = None # Ex component at the XZ plane
        Ey_XZ: Optional[Matrix] = None # Ey component at the XZ plane
        Ez_XZ: Optional[Matrix] = None # Ez component at the XZ plane
        Ex_XY: Optional[Matrix] = None # Ex component at the XY plane
        Ey_XY: Optional[Matrix] = None # Ey component at the XY plane
        Ez_XY: Optional[Matrix] = None # Ez component at the XY plane
        
        def calculate_intensity(self):
            self.Intensity_XZ = np.abs(self.Ex_XZ)**2+np.abs(self.Ey_XZ)**2+np.abs(self.Ez_XZ)**2
            self.Intensity_XY = np.abs(self.Ex_XY)**2+np.abs(self.Ey_XY)**2+np.abs(self.Ez_XY)**2
        
        def calculate_intensity_along_x(self):
            self.Intensity_along_x=self.Intensity_XY[int(np.shape(self.Intensity_XY)[0]/2),:]

    @dataclass
    class FieldAtFocus3D:
        """Field near the focus. Contains the 3 components and the intensity. 
        The positions of the matrix are given by: field[Nz, Ny, Nx] with Nz, Ny, Nx the index of the positions where the field was calculated for Z, Y, X"""
        Ex: Optional[Matrix3D] = None # Ex component
        Ey: Optional[Matrix3D] = None # Ey component
        Ez: Optional[Matrix3D] = None # Ez component
        
        def calculate_intensity(self):
            self.Intensity = np.abs(self.Ex)**2+np.abs(self.Ey)**2+np.abs(self.Ez)**2
        
        def calculate_intensity_along_x(self, axial_position: int):
            self.Intensity_along_x=self.Intensity[axial_position,int(np.shape(self.Intensity)[1]/2),:]
    
    class FocusFieldParameters(BaseModel):
        '''Parameters for the simulation of the field near the focus'''
        NA: Union[StrictFloat, StrictInt] # Numerical aperture
        n: Union[StrictFloat, StrictInt, Any] # Refraction index for the medium of the optical system. 
        h: Union[StrictFloat, StrictInt] # Radius of aperture of the objective lens
        f: Union[StrictFloat, StrictInt] = 0 # Focal distance
        
        x_steps: Union[StrictFloat, StrictInt] # Resolution in the X or Y coordinate for the focused field (nm)
        z_steps: Union[StrictFloat, StrictInt] # Resolution in the axial coordinate (Z) for the focused field (nm)
        x_range: Union[StrictFloat, StrictInt] # Field of view in the X or Y coordinate in which the focused field is calculated (nm)
        z_range: Union[StrictFloat, StrictInt] # Field of view in the axial coordinate (z) in which the focused field is calculated (nm)
        
        z: Union[StrictFloat, StrictInt] = 0 # Axial position for the XY plane (nm)
        phip: Union[StrictFloat, StrictInt] = 0 # Angle of rotation along the z axis when calculating the field
        x0: Union[StrictFloat, StrictInt] = 0 # Position along X around wich to calculate the fielad near the focus (nm) 
        y0: Union[StrictFloat, StrictInt] = 0 # Position along Y around wich to calculate the fielad near the focus (nm)
        
        field_parameters: FieldParameters
        custom_mask_parameters: CustomMaskParameters = CustomMaskParameters()
        interface_parameters: Optional[InterfaceParameters] = None
        
        @property
        def alpha(self) -> float:
            if self.interface_parameters is None:
                return np.arcsin(self.NA / self.n)
            else:
                return np.arcsin(self.NA / self.interface_parameters.ns[0])
        
        @property
        def z_step_count(self) -> int:
            '''Number of steps along the axial coordinate'''
            return int(np.rint(self.z_range/self.z_steps/2))
        
        @property
        def r_step_count(self) -> int:
            '''Number of steps along the radial coordinate'''
            return int(np.rint(self.x_range/self.x_steps/2**0.5)) #the actual field of view of the X axis in the XZ plane will be x_range*2**0.5

        
        def transform_input_parameter_units(self):
            '''transforms units from degrees to radians and from milimeters to nanometers'''
            if self.interface_parameters is None:
                self.field_parameters.wavelength /= self.n
            self.field_parameters.transform_input_parameter_units()
            
            if isinstance(self.n, (int,float)): # Compatibility with pyfocus<3.0
                self.f = self.h * self.n/ self.NA *10**6
            else:
                self.f = self.h * self.n[0]/ self.NA *10**6
            
            #transform to radians:
            self.phip /= 180*np.pi

    @abstractclassmethod
    def calculate(focus_field_parameters: FocusFieldParameters) -> FieldAtFocus:
        """Main function that calculates the field along the XY and XZ planes. Orchestrates the steps of the calculation.

        Args:
            :focus_field_parameters (FocusFieldCalculator.FocusFieldParameters): Parameters of the simulation
            :maskfunction (Callable): Analytical function that defines the phase mask, must be a function of the 5 internal variables: rho, phi, w0, f and k, with:
                
                rho: Radial coordinate from 0 to the aperture radius of the objective.
                
                phi: Azimutal coordinate from 0 to 2pi.
                
                w0: Radius of the incident gaussian beam.
                
                f: Focal distane of the objective lens (mm)
                
                k: Wavenumber in the objective lens medium (mm)
                
                The real part defines the amplitude of the incident field

        Returns:
            FocusFieldCalculator.FieldAtFocus: Field near the focus on a plane. Contains the 3 components and the intensity
        """
        raise NotImplementedError
    
    def _calculate_matrix_size(self, x_range: int, x_steps: int) -> Tuple[int, int]:
        return int(np.rint(x_range/x_steps/2-1)*2), int(np.rint(x_range/x_steps/2-1)*2)
    
    def _initialize_fields(self, x_size: int, y_size: int) -> List[list]:
        '''Returns empty matrixes to fill with the values of the field'''
        return [np.zeros((x_size, y_size),dtype=complex) for _ in range(6)]

    def _mirror_on_z_axis(self, matrixes):
        '''Mirrors the matrixes with the values of the field along the z axis'''
        output = []
        for matrix in deepcopy(matrixes):
            output.append(np.vstack((np.flipud(np.conj(matrix)),matrix[1:,:])))
        return output
    
    def _integrate(self, matrix_ammount, functions_to_integrate, focus_field_parameters: FocusFieldParameters, description):
        '''Performs the numerical integration of the functions "functions_to_integrate" at different positions of z and r'''
        matrixes = [np.zeros((focus_field_parameters.z_step_count, focus_field_parameters.r_step_count),dtype=complex) for _ in range(matrix_ammount)]
        
        for n_z in tqdm(range(focus_field_parameters.z_step_count),desc=description): #TODO ir ploteando los valores que toma n_z
            for n_r in range(focus_field_parameters.r_step_count):
                kz=n_z*2*np.pi/focus_field_parameters.field_parameters.wavelength/focus_field_parameters.z_step_count*focus_field_parameters.z_range/2 
                kr=n_r*2*np.pi/focus_field_parameters.field_parameters.wavelength/focus_field_parameters.r_step_count*focus_field_parameters.x_range/2*2**0.5
                
                for i, matrix in enumerate(matrixes):
                    result = complex_quadrature(functools.partial(functions_to_integrate[i], kz=kz, kr=kr),0,focus_field_parameters.alpha)[0]
                    (matrix)[n_z,n_r] = result
        
        return matrixes
    
    def _calculate_amplitude_factors(self, parameters: FocusFieldParameters): #TODO actualziar parametros en llamada a funcion
        """Calculates the base amplitude factors used in the calculation"""
        E1=np.sqrt(parameters.field_parameters.I_0)*np.cos(parameters.field_parameters.polarization.gamma)/parameters.field_parameters.wavelength*np.pi*parameters.f
        E2=np.sqrt(parameters.field_parameters.I_0)*np.sin(parameters.field_parameters.polarization.gamma)/parameters.field_parameters.wavelength*np.pi*parameters.f*np.exp(1j*parameters.field_parameters.polarization.beta)
        return E1, E2
    
    @abstractclassmethod
    def calculate_3D_field(self, *args, **kwargs) -> FieldAtFocus3D:
        """Main function that calculates the field along a 3D space by calculating it as slices of 2D XY planes. 
        Orchestrates the steps of the calculation.

        Args:
            :focus_field_parameters (FocusFieldCalculator.FocusFieldParameters): Parameters of the simulation
            :maskfunction (Callable): Analytical function that defines the phase mask, must be a function of the 5 internal variables: rho, phi, w0, f and k, with:
                
                rho: Radial coordinate from 0 to the aperture radius of the objective.
                
                phi: Azimutal coordinate from 0 to 2pi.
                
                w0: Radius of the incident gaussian beam.
                
                f: Focal distane of the objective lens (mm)
                
                k: Wavenumber in the objective lens medium (mm)
                
                The real part defines the amplitude of the incident field

        Returns:
            FocusFieldCalculator.FieldAtFocus3D: Field near the focus. Contains the 3 components and the intensity. 
            The positions of the matrix are given by: field[Nz, Ny, Nx] with Nz, Ny, Nx the index of the positions where the field was calculated for Z, Y, X
            For more information read the corresponding dataclass
        """
        raise NotImplementedError
