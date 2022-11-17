import pytest 
from src.model.focus_field_calculators.base import FocusFieldCalculator
from src.model.free_propagation_calculators.base import FreePropagationCalculator
from src.custom_dataclasses.field_parameters import FieldParameters, PolarizationParameters
from src.custom_dataclasses.mask import MaskType

from src.model.main_calculation_handler import MainCalculationHandler

def create_base_parameters(base_simulation_parameters=None, field_parameters=None, polarization=None, lens_parameters=None, focus_parameters=None):
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
        field_parameters = FieldParameters(w0=10, wavelength=500, I_0=1, polarization=polarization)
    
    if not lens_parameters: 
        lens_parameters = FreePropagationCalculator.ObjectiveFieldParameters(L=50, R=10, field_parameters=field_parameters)
    if not focus_parameters: 
        focus_parameters = FocusFieldCalculator.FocusFieldParameters(NA=1.4, n=1.5, h=3, f=100, z=0, x_steps=20, z_steps=10, x_range=1000, z_range=1000, phip=0, field_parameters=field_parameters)
    
    return base_simulation_parameters, lens_parameters, focus_parameters


def test_default_mask_focus_field():
    base_simulation_parameters, lens_parameters, focus_parameters = create_base_parameters(
        base_simulation_parameters = MainCalculationHandler.BasicParameters(
                file_name='test_default_mask_focus_field', 
                propagate_incident_field=False,
                plot_incident_field=False, 
                plot_focus_field_amplitude=True,
                plot_focus_field_intensity=True
            )
        )
    
    calculation_handler = MainCalculationHandler(strategy=MaskType.no_mask)
    field = calculation_handler.calculate_field(base_simulation_parameters, lens_parameters, focus_parameters)


def test_default_mask_objective_field():
    ...
    
def test_custom_mask_focus_field():
    ...

def test_custom_mask_objective_field():
    ...
    
def test_interface_custom_mask_focus_field():
    ...

def test_interface_default_mask_focus_field():
    ...

