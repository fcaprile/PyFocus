"""This file is for compatibility with PyFocus<3.0

It works as an adapter with the reworked logic and the way the functions were called for PyFocus<3.0
"""

from .log_config import logger
from .model.focus_field_calculators.base import FocusFieldCalculator
from .model.free_propagation_calculators.base import FreePropagationCalculator
from .custom_dataclasses.field_parameters import FieldParameters, PolarizationParameters
from .custom_dataclasses.custom_mask import CustomMaskParameters
from .custom_dataclasses.mask import MaskType
from .custom_dataclasses.interface_parameters import InterfaceParameters
from .model.main_calculation_handler import MainCalculationHandler
import numpy as np

compatibility_error_msg = " is no longer supported for PyFocus>=3.0. For any questions feel free to contact us via mail or on github!"

def create_parameters(propagation=False,multilayer=False,NA=1.4,n=1.5,h=3,w0=5,wavelength=640,gamma=45,beta=90,z=0,x_steps=5,z_steps=8,x_range=1000,z_range=2000,I0=1,L='',R='',ds='',z_int='',figure_name='',divisions_theta=200,divisions_phi=200,plot_Ei=True):
    if propagation is True: raise NotImplementedError("Propagation"+compatibility_error_msg)
    
    base_simulation_parameters = MainCalculationHandler.BasicParameters(
        file_name=figure_name, 
        propagate_incident_field=False,
        plot_incident_field=plot_Ei, 
        plot_focus_field_amplitude=True,
        plot_focus_field_intensity=True
    )
    
    polarization = PolarizationParameters(gamma=gamma, beta=beta)
    
    field_parameters = FieldParameters(w0=w0, wavelength=wavelength, I_0=I0, polarization=polarization)
    
    lens_parameters = FreePropagationCalculator.ObjectiveFieldParameters(L=L, R=R, field_parameters=field_parameters)
    
    if multilayer is False:
        interface_parameters = None
    else:
        interface_parameters = InterfaceParameters(axial_position=z_int, ns=n, ds=ds)

    
    focus_parameters = FocusFieldCalculator.FocusFieldParameters(NA=NA, n=n, h=h, x_steps=x_steps, z_steps=z_steps, x_range=x_range, z_range=z_range, z=z, phip=0, field_parameters=field_parameters, interface_parameters=interface_parameters, custom_mask_parameters=CustomMaskParameters(divisions_theta, divisions_phi))    
    
    return base_simulation_parameters, lens_parameters, focus_parameters



def no_mask(propagation=False,multilayer=False,NA=1.4,n=1.5,h=3,w0=5,wavelength=640,gamma=45,beta=90,z=0,x_steps=5,z_steps=8,x_range=1000,z_range=2000,I0=1,L='',R='',ds='',z_int='',figure_name=''):#f (the focal distance) is given by the sine's law, but can be modified if desired
    base_simulation_parameters, lens_parameters, focus_parameters = create_parameters(propagation,multilayer,NA,n,h,w0,wavelength,gamma,beta,z,x_steps,z_steps,x_range,z_range,I0,L,R,ds,z_int,figure_name)
    calculation_handler = MainCalculationHandler(mask_type=MaskType.no_mask)
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters)
    return field.Ex_XZ, field.Ey_XZ, field.Ez_XZ, field.Ex_XY, field.Ey_XY, field.Ez_XY

def VP(propagation=False,multilayer=False,NA=1.4,n=1.5,h=3,w0=5,wavelength=640,gamma=45,beta=90,z=0,x_steps=5,z_steps=8,x_range=1000,z_range=2000,I0=1,L='',R='',ds='',z_int='',figure_name=''):
    base_simulation_parameters, lens_parameters, focus_parameters = create_parameters(propagation,multilayer,NA,n,h,w0,wavelength,gamma,beta,z,x_steps,z_steps,x_range,z_range,I0,L,R,ds,z_int,figure_name)
    calculation_handler = MainCalculationHandler(mask_type=MaskType.vortex_mask)
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters)
    return field.Ex_XZ, field.Ey_XZ, field.Ez_XZ, field.Ex_XY, field.Ey_XY, field.Ez_XY

def custom(entrance_field, custom_mask, propagation=False,multilayer=False,NA=1.4,n=1.5,h=3,w0=5,wavelength=640,gamma=45,beta=90,z=0,x_steps=5,z_steps=8,x_range=1000,z_range=2000,I0=1,L='',R='',ds='',z_int='',figure_name='',divisions_theta=200,divisions_phi=200,plot_Ei=True):#f (the focal distance) is given by the sine's law, but can be modified if desired
    base_simulation_parameters, lens_parameters, focus_parameters = create_parameters(propagation,multilayer,NA,n,h,w0,wavelength,gamma,beta,z,x_steps,z_steps,x_range,z_range,I0,L,R,ds,z_int,figure_name,divisions_theta,divisions_phi,plot_Ei)
    calculation_handler = MainCalculationHandler(mask_type=MaskType.custom_mask)
    custom_field_function=lambda rho, phi,w0,f,k: entrance_field(rho, phi,w0,f,k)* custom_mask(rho, phi,w0,f,k)
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters, mask_function=custom_field_function)
    return field.Ex_XZ, field.Ey_XZ, field.Ez_XZ, field.Ex_XY, field.Ey_XY, field.Ez_XY
