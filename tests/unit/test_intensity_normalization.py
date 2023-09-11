import logging
import numpy as np
import pytest 
import os
import sys

sys.path.append(os.path.abspath("../../"))

from src.PyFocus.model.main_calculation_handler import MainCalculationHandler
from src.PyFocus.log_config import logger
from src.PyFocus.custom_dataclasses.mask import MaskType


def test_normalization_of_uniform_beam():
    logger.setLevel(logging.INFO)
    do_a_precise_simulation_and_plot_it = False
    
    calculation_handler = MainCalculationHandler(mask_type=MaskType.custom_mask)
    lens_apperture = 3
    mask_function = lambda rho, phi, w0,f,k : 1
    ratio = calculation_handler.calculate_incident_energy_ratio(lens_apperture, mask_function, None, None, None)
    assert ratio == 1

def test_normalization_of_custom_beam():
    logger.setLevel(logging.INFO)
    do_a_precise_simulation_and_plot_it = False
    
    calculation_handler = MainCalculationHandler(mask_type=MaskType.custom_mask)
    lens_apperture = 3
    mask_function = lambda rho, phi, w0,f,k : (rho)**0.5*np.exp(1j*phi)
    ratio = calculation_handler.calculate_incident_energy_ratio(lens_apperture, mask_function, None, None, None)
    print(ratio)