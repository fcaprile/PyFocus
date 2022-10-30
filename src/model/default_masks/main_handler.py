

from dataclasses import dataclass
from typing import Dict, Tuple, Type
from dataclass.mask import MaskType

import numpy as np

from model.default_masks.focus_field_calculators.rotation_simmetry_mask import RotationSimmetryFocusFieldCalculator
from model.default_masks.focus_field_calculators.no_mask import NoMaskFocusFieldCalculator
from model.default_masks.focus_field_calculators.vortex_mask import VortexMaskFocusFieldCalculator
from model.default_masks.free_propagation_calculators.vortex_mask import VortexMaskFreePropagationCalculator
from model.default_masks.free_propagation_calculators.no_mask import NoMaskFreePropagationCalculator
from src.equations.helpers.plot.plot_objective_field import plot_objective_field
from src.model.default_masks.focus_field_calculators.base import DefaultMaskFocusFieldCalculator
from src.model.default_masks.free_propagation_calculators.base import DefaultMaskFreePropagationCalculator
from src.model.default_masks.free_propagation_calculators.rotation_simmetry_mask import RotationSimmetryFreePropagationCalculator

class DefaultMaskCalculationMainHandler:
    _STRATEGY_MAPPER: Dict[MaskType, Tuple[Type[DefaultMaskFreePropagationCalculator], Type[DefaultMaskFocusFieldCalculator]]] = {
        MaskType.no_mask: (NoMaskFreePropagationCalculator, NoMaskFocusFieldCalculator),
        MaskType.vortex_mask: (VortexMaskFreePropagationCalculator, VortexMaskFocusFieldCalculator),
        MaskType.rotation_simmetry: (RotationSimmetryFreePropagationCalculator, RotationSimmetryFocusFieldCalculator)
    }
    
    @dataclass
    class CalculationParameters:
        f: float
        
        def transform_input_parameter_units(self): # TODO Que sea algo propio de cada clase abstracta, para tener una transformacion en el objetivo y otra en el foco
            #transform to nanometers
            self.f*=10**6
            self.w0*=10**6
            
            #transform to radians:
            self.beta *= np.pi/180
            self.phip /= 180*np.pi
            self.gamma *= np.pi/180
    
    def __init__(self, strategy: MaskType, plot_function: callable = None) -> None:
        self._objective_field_calculator, self._focus_field_calculator = self._STRATEGY_MAPPER[strategy]
        
        self.plot_objective_field_function = plot_function if plot_function else plot_objective_field
        
    def calculate_field(self, calculation_parameters: dict):
        _parameters = self.CalculationParameters(calculation_parameters)
        _parameters.transform_input_parameter_units()
        
        objective_field = self._objective_field_calculator().execute(_parameters)
        
        if calculation_parameters.plot==True:
            self.plot_objective_field_function(*objective_field, xmax=_parameters.xmax, figure_name=_parameters.figure_name)
        
        focus_field = self._focus_field_calculator(_parameters).execute(_parameters)