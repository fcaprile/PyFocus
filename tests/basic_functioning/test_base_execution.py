import logging
import pytest 
import os
import sys

from .plot_2D import plot_along_z_and_x

sys.path.append(os.path.abspath("../../src"))

from PyFocus.log_config import logger
from PyFocus.model.focus_field_calculators.base import FocusFieldCalculator
from PyFocus.model.free_propagation_calculators.base import FreePropagationCalculator
from PyFocus.custom_dataclasses.field_parameters import FieldParameters, PolarizationParameters
from PyFocus.custom_dataclasses.mask import MaskType
from PyFocus.custom_dataclasses.interface_parameters import InterfaceParameters
from PyFocus.model.main_calculation_handler import MainCalculationHandler
import numpy as np


def create_base_parameters(base_simulation_parameters=None, field_parameters=None, polarization=None, lens_parameters=None, focus_parameters=None, interface_parameters=None, precise_simulation:bool = False):
    if not base_simulation_parameters:
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
            file_name='test', 
            propagate_incident_field=False,
            plot_incident_field=False, 
            plot_focus_field_amplitude=False,
            plot_focus_field_intensity=False
        )
    
    if not field_parameters:
        if not polarization:
            polarization = PolarizationParameters(gamma=45, beta=90)
        field_parameters = FieldParameters(w0=50, wavelength=640, I_0=1, polarization=polarization)
    
    if not lens_parameters: 
        lens_parameters = FreePropagationCalculator.ObjectiveFieldParameters(L=50, R=10, field_parameters=field_parameters)
    
    if not focus_parameters:
        if precise_simulation: 
            x_steps=10
            z_steps=20
        else:
            x_steps=300
            z_steps=600
        focus_parameters = FocusFieldCalculator.FocusFieldParameters(NA=1.4, n=1.5, h=3, x_steps=x_steps, z_steps=z_steps, x_range=1000, z_range=2000, z=0, phip=0, field_parameters=field_parameters, interface_parameters=interface_parameters)    
    
    return base_simulation_parameters, lens_parameters, focus_parameters


def test_no_mask_focus_field():
    logger.setLevel(logging.INFO)
    do_a_precise_simulation_and_plot_it = False
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_no_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=do_a_precise_simulation_and_plot_it,
                plot_focus_field_intensity=do_a_precise_simulation_and_plot_it
            ),
        precise_simulation=do_a_precise_simulation_and_plot_it
        )
    
    calculation_handler = MainCalculationHandler(mask_type=MaskType.no_mask)
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters)

def test_with_napari_parameters():
    logger.setLevel(logging.INFO)
    do_a_precise_simulation_and_plot_it = False
    polarization = PolarizationParameters(gamma=45, beta=90)
    field_parameters = FieldParameters(w0=50, wavelength=532*1.33, I_0=1, polarization=polarization)
    focus_parameters = FocusFieldCalculator.FocusFieldParameters(NA=0.65, n=1.33, h=3, x_steps=30, z_steps=200, x_range=1500, z_range=1200, z=0, phip=0, field_parameters=field_parameters, interface_parameters=None)
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_no_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=do_a_precise_simulation_and_plot_it,
                plot_focus_field_intensity=do_a_precise_simulation_and_plot_it
            ),
        precise_simulation=do_a_precise_simulation_and_plot_it,
        focus_parameters=focus_parameters
        )
    
    calculation_handler = MainCalculationHandler(mask_type=MaskType.custom_mask)
    mask_function = lambda rho, phi,w0,f,k: 1
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters, mask_function=mask_function)

def test_no_mask_focus_field_with_x_polarization():
    logger.setLevel(logging.INFO)
    do_a_precise_simulation_and_plot_it = False
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_no_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=do_a_precise_simulation_and_plot_it,
                plot_focus_field_intensity=do_a_precise_simulation_and_plot_it
            ),
        precise_simulation=do_a_precise_simulation_and_plot_it,
        polarization = PolarizationParameters(gamma=0, beta=0)
        )
    
    calculation_handler = MainCalculationHandler(mask_type=MaskType.no_mask)
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters)


def test_VP_mask_focus_field():
    logger.setLevel(logging.INFO)
    do_a_precise_simulation_and_plot_it = False
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_VP_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=do_a_precise_simulation_and_plot_it,
                plot_focus_field_intensity=do_a_precise_simulation_and_plot_it
            ),
        precise_simulation=do_a_precise_simulation_and_plot_it
        )
    
    calculation_handler = MainCalculationHandler(mask_type=MaskType.vortex_mask)
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters)
    
def test_custom_mask_focus_field():
    logger.setLevel(logging.INFO)
    do_a_precise_simulation_and_plot_it = False
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_custom_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=do_a_precise_simulation_and_plot_it,
                plot_focus_field_intensity=do_a_precise_simulation_and_plot_it
            ),
        precise_simulation=do_a_precise_simulation_and_plot_it
        )
    
    calculation_handler = MainCalculationHandler(mask_type=MaskType.custom_mask)
    mask_function = lambda rho, phi,w0,f,k: np.exp(1j*phi) 
    # mask_function = lambda rho, phi,w0,f,k: 1
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters, mask_function=mask_function)

def test_interface_custom_mask_focus_field():
    logger.setLevel(logging.INFO)
    do_a_precise_simulation_and_plot_it = False
    interface_parameters = InterfaceParameters(axial_position=0, ns=np.array((1.5,1.5)), ds=np.array((np.inf,np.inf)))
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_interface_custom_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=do_a_precise_simulation_and_plot_it,
                plot_focus_field_intensity=do_a_precise_simulation_and_plot_it,
            ),
        precise_simulation=do_a_precise_simulation_and_plot_it,
        interface_parameters=interface_parameters
        )
    
    calculation_handler = MainCalculationHandler(mask_type=MaskType.custom_mask)
    
    mask_function = lambda rho, phi,w0,f,k: np.exp(1j*phi) 
    #mask_function = lambda rho, phi,w0,f,k: 1
    field = calculation_handler.calculate_2D_fields(base_simulation_parameters, lens_parameters, focus_parameters, mask_function=mask_function)

def test_interface_default_mask_focus_field():
    ...

def test_3D_field_custom_mask(): # Checked values with previous pyfocus version
    logger.setLevel(logging.DEBUG)
    plot = False
    polarization = PolarizationParameters(gamma=45, beta=90)
    wavelength_0 = 532
    dr = 30
    dz = 200
    Nxy = 1
    Nz = 1
    field_parameters = FieldParameters(w0=50, wavelength=wavelength_0, I_0=1, polarization=polarization)
    focus_parameters = FocusFieldCalculator.FocusFieldParameters(NA=0.65, n=1.5, h=3, x_steps=dr, z_steps=dz, x_range=dr*Nxy*2**0.5, z_range=dz*Nz*2, z=0, phip=0, field_parameters=field_parameters, interface_parameters=None)
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_3D_field_custom_mask', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=False,
                plot_focus_field_intensity=False
            ),
        precise_simulation=False,
        focus_parameters=focus_parameters
        )
    
    calculation_handler = MainCalculationHandler(mask_type=MaskType.custom_mask)
    mask_function = lambda rho, phi,w0,f,k: 1
    # mask_function = lambda rho, phi,w0,f,k: np.exp(1j*phi) 
    field = calculation_handler.calculate_3D_fields(base_simulation_parameters, lens_parameters, focus_parameters, mask_function=mask_function)
    if plot: plot_along_z_and_x(field=field, focus_parameters=focus_parameters, wavelength_0=wavelength_0, dr=dr, dz=dz, Nxy=Nxy)


def test_3D_field_custom_mask_with_interface():
    logger.setLevel(logging.DEBUG)
    plot = True
    interface_parameters = InterfaceParameters(axial_position=0, ns=np.array((1.5,1.5)), ds=np.array((np.inf,np.inf)))
    polarization = PolarizationParameters(gamma=45, beta=90)
    wavelength_0 = 532
    dr = 30
    dz = 200
    Nxy = 31
    Nz = 11
    field_parameters = FieldParameters(w0=50, wavelength=wavelength_0, I_0=1, polarization=polarization)
    focus_parameters = FocusFieldCalculator.FocusFieldParameters(NA=0.65, n=1.5, h=3, x_steps=dr, z_steps=dz, x_range=dr*Nxy*2**0.5, z_range=dz*Nz*2, z=0, phip=0, field_parameters=field_parameters, interface_parameters=interface_parameters)
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_3D_field_custom_mask', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=False,
                plot_focus_field_intensity=True
            ),
        precise_simulation=False,
        focus_parameters=focus_parameters,
        interface_parameters=interface_parameters
        )
    
    calculation_handler = MainCalculationHandler(mask_type=MaskType.custom_mask)
    mask_function = lambda rho, phi,w0,f,k: 1
    # mask_function = lambda rho, phi,w0,f,k: np.exp(1j*phi) 
    field = calculation_handler.calculate_3D_fields(base_simulation_parameters, lens_parameters, focus_parameters, mask_function=mask_function)
    if plot: plot_along_z_and_x(field=field, focus_parameters=focus_parameters, wavelength_0=wavelength_0, dr=dr, dz=dz, Nxy=Nxy)
