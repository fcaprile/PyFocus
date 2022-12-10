import pytest 
import os
import sys
sys.path.append(os.path.abspath("./src"))

from model.focus_field_calculators.base import FocusFieldCalculator
from model.free_propagation_calculators.base import FreePropagationCalculator
from custom_dataclasses.field_parameters import FieldParameters, PolarizationParameters
from custom_dataclasses.mask import MaskType

from model.main_calculation_handler import MainCalculationHandler

def create_base_parameters(base_simulation_parameters=None, field_parameters=None, polarization=None, lens_parameters=None, focus_parameters=None, precise_simulation:bool = False):
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
            focus_parameters = FocusFieldCalculator.FocusFieldParameters(NA=1.4, n=1.5, h=3, x_steps=10, z_steps=20, x_range=1000, z_range=2000, z=0, phip=0, field_parameters=field_parameters)
        else:
            focus_parameters = FocusFieldCalculator.FocusFieldParameters(NA=1.4, n=1.5, h=3, x_steps=300, z_steps=600, x_range=1000, z_range=2000, z=0, phip=0, field_parameters=field_parameters)    
    
    return base_simulation_parameters, lens_parameters, focus_parameters


def test_no_mask_focus_field():
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_no_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=False,
                plot_focus_field_intensity=False
            ),
        precise_simulation=False
        )
    
    calculation_handler = MainCalculationHandler(strategy=MaskType.no_mask)
    field = calculation_handler.calculate_field(base_simulation_parameters, lens_parameters, focus_parameters)


def test_VP_mask_focus_field():
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_VP_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=False,
                plot_focus_field_intensity=False
            ),
        precise_simulation=False
        )
    
    calculation_handler = MainCalculationHandler(strategy=MaskType.vortex_mask)
    field = calculation_handler.calculate_field(base_simulation_parameters, lens_parameters, focus_parameters)
    
def test_custom_mask_focus_field():
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_custom_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=True,
                plot_focus_field_intensity=True
            ),
        precise_simulation=True
        )
    
    calculation_handler = MainCalculationHandler(strategy=MaskType.custom_mask)
    import numpy as np
    mask_function = lambda rho, phi,w0,f,k: 1#np.exp(1j*phi) 
    field = calculation_handler.calculate_field(base_simulation_parameters, lens_parameters, focus_parameters, mask_function=mask_function)

def test_custom_mask_objective_field():
    ...
    
def test_interface_custom_mask_focus_field():
    ...

def test_interface_default_mask_focus_field():
    ...

