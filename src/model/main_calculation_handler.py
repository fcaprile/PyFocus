import traceback
from typing import Dict, Tuple, Type
from custom_dataclasses.mask import MaskType
from pydantic import BaseModel, StrictBool, StrictStr

from model.focus_field_calculators.propagated_vortex_mask import PropagatedVortexMaskFocusFieldCalculator
from model.focus_field_calculators.no_mask import NoMaskFocusFieldCalculator
from model.focus_field_calculators.vortex_mask import VortexMaskFocusFieldCalculator
from model.focus_field_calculators.base import FocusFieldCalculator

from model.free_propagation_calculators.vortex_mask import VortexMaskFreePropagationCalculator
from model.free_propagation_calculators.no_mask import NoMaskFreePropagationCalculator
from model.free_propagation_calculators.vortex_mask import VortexMaskFreePropagationCalculator
from model.free_propagation_calculators.base import FreePropagationCalculator
from plot_functions import PlotParameters

from plot_functions.plot_objective_field import plot_objective_field
from plot_functions.plot_at_focus import plot_amplitude_and_phase_at_focus, plot_intensity_at_focus

class MainCalculationHandler:
    _STRATEGY_MAPPER: Dict[MaskType, Tuple[Type[FreePropagationCalculator], Type[FocusFieldCalculator]]] = {
        MaskType.no_mask: (NoMaskFreePropagationCalculator, NoMaskFocusFieldCalculator),
        MaskType.vortex_mask: (VortexMaskFreePropagationCalculator, VortexMaskFocusFieldCalculator),
        MaskType.propagated_vortex_mask: (VortexMaskFreePropagationCalculator, PropagatedVortexMaskFocusFieldCalculator)
    }
    
    class BasicParameters(BaseModel):
        '''Base configuration for the whole simulation''' # TODO comentar aca que es cada parametro
        file_name: StrictStr
        
        propagate_incident_field: StrictBool
        plot_incident_field: StrictBool
        incident_field_figure_name: StrictStr = 'Intensity of the incident field'
        
        plot_focus_field_intensity: StrictBool
        focus_field_intensity_figure_name: StrictStr = 'Intensity at the focus'
        
        plot_focus_field_amplitude: StrictBool 
        focus_field_amplitude_figure_name: StrictStr = 'Amplitude at the focus'
        

    def __init__(self, strategy: MaskType) -> None:
        _objective_field_calculator, _focus_field_calculator = self._STRATEGY_MAPPER[strategy]
        self._objective_field_calculator = _objective_field_calculator()
        self._focus_field_calculator = _focus_field_calculator()
        
        self.plot_objective_field_function = plot_objective_field
    
    def _handle_propagated_field_calculation(self, basic_parameters, objective_field_parameters):
        E_rho,Ex,Ey = self._objective_field_calculator()
        if basic_parameters.plot_incident_field == True:
            self.plot_objective_field_function(Ex,Ey, objective_field_parameters)
        fields = self._focus_field_calculator(E_rho)
    
    def calculate_field(
            self, 
            basic_parameters: BasicParameters, 
            objective_field_parameters: FreePropagationCalculator.ObjectiveFieldParameters, 
            focus_field_parameters: FocusFieldCalculator.FocusFieldParameters
        ):
        if basic_parameters.propagate_incident_field:
            objective_field_parameters.transform_input_parameter_units()
            return self._handle_propagated_field_calculation(basic_parameters, objective_field_parameters)        
        
        focus_field_parameters.field_parameters.wavelength /= focus_field_parameters.n # TODO manejar diferencia multicapa o no
        focus_field_parameters.transform_input_parameter_units()
        focus_field = self._focus_field_calculator.calculate(focus_field_parameters)
        if basic_parameters.plot_focus_field_intensity == True:
            plot_params=PlotParameters(name=basic_parameters.focus_field_intensity_figure_name)
            plot_intensity_at_focus(focus_field, focus_field_parameters, params=plot_params) #TODO añadir el size
        if basic_parameters.plot_focus_field_amplitude == True:
            plot_params=PlotParameters(name=basic_parameters.focus_field_amplitude_figure_name)
            plot_amplitude_and_phase_at_focus(focus_field, focus_field_parameters, params=plot_params)
        
        return focus_field
        