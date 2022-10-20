

from dataclasses import dataclass
from model.default_mask.base import DefaultMaskHandler
from model.default_mask.VP_functions import VortexMaskHandler
from model.default_mask.no_mask_functions import NoMaskHandler
from dataclass.mask import MaskType

import numpy as np

class DefaultMaskCalculationMainHandler:
    _STRATEGY_MAPPER = {
        MaskType.no_mask: NoMaskHandler,
        MaskType.vortex_mask: VortexMaskHandler
    }
    
    @dataclass
    class CalculationParameters:
        f: float
        
        def transform_input_parameter_units(self):
            #transform to nanometers
            self.f*=10**6
            self.w0*=10**6
            
            #transform to radians:
            self.beta *= np.pi/180
            self.phip /= 180*np.pi
            self.gamma *= np.pi/180
    
    def __init__(self, strategy: MaskType) -> None:
        self._handler: DefaultMaskHandler = self._STRATEGY_MAPPER[strategy]
    
    def calculate_field(self, calculation_parameters: dict):
        parameters = self.CalculationParameters(calculation_parameters)
        parameters.transform_input_parameter_units()
        
        field = self._handler.calculate_field()
        