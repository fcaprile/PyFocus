import traceback
import numpy as np
from typing import Dict, Tuple, Type
from ..custom_dataclasses.mask import MaskType
from pydantic import BaseModel, StrictBool, StrictStr
from matplotlib import pyplot as plt

from ..model.focus_field_calculators.propagated_vortex_mask import PropagatedVortexMaskFocusFieldCalculator
from ..model.focus_field_calculators.no_mask import NoMaskFocusFieldCalculator
from ..model.focus_field_calculators.vortex_mask import VortexMaskFocusFieldCalculator
from ..model.focus_field_calculators.base import FocusFieldCalculator
from ..model.focus_field_calculators.custom_mask import CustomMaskFocusFieldCalculator
from ..model.free_propagation_calculators.custom_mask import CustomMaskFreePropagationCalculator

from ..model.free_propagation_calculators.vortex_mask import VortexMaskFreePropagationCalculator
from ..model.free_propagation_calculators.no_mask import NoMaskFreePropagationCalculator
from ..model.free_propagation_calculators.vortex_mask import VortexMaskFreePropagationCalculator
from ..model.free_propagation_calculators.base import FreePropagationCalculator

from ..log_config import logger

from ..plot_functions import PlotParameters
from ..plot_functions.plot_objective_field import plot_objective_field
from ..plot_functions.plot_at_focus import plot_amplitude_and_phase_at_focus, plot_intensity_at_focus
from ..plot_functions.plot_at_focus_3D import plot_3D_amplitude_and_phase_at_focus, plot_3D_intensity_at_focus

class MainCalculationHandler:
    """Main class that orchestrates a simulation. 
    It chooses which class and function to use to perform the numerical integration and calls a function to plot the resulting field.
    It also handles the conversion between different units of measurement (mm to nm, degrees to radians, etc...)
    
    When instanciated, a mask type (No mask, VP mask or custom mask) must be given so that it selects the calculation needed
    Later, one of the functions calculate_2D_fields or calculate_3D_fields must be called to calculate the resulting field
    """
    
    _STRATEGY_MAPPER: Dict[MaskType, Tuple[Type[FreePropagationCalculator], Type[FocusFieldCalculator], bool]] = {
        MaskType.no_mask: (NoMaskFreePropagationCalculator, NoMaskFocusFieldCalculator, True),
        MaskType.vortex_mask: (VortexMaskFreePropagationCalculator, VortexMaskFocusFieldCalculator, True),
        MaskType.custom_mask: (CustomMaskFreePropagationCalculator, CustomMaskFocusFieldCalculator, False),
        MaskType.propagated_vortex_mask: (VortexMaskFreePropagationCalculator, PropagatedVortexMaskFocusFieldCalculator, True),
    }
    
    class BasicParameters(BaseModel):
        '''Basic configuration for the simulation'''
        file_name: StrictStr # Name by wich to save the file containing the field's amplitude TODO not implemented yet
        
        propagate_incident_field: StrictBool # Wheter or not to calculate the field's propagation to the lens TODO if True, not implemented yet
        plot_incident_field: StrictBool # Wheter or not to plot the field inciding at the lens
        incident_field_figure_name: StrictStr = 'Intensity of the incident field'
        
        plot_focus_field_intensity: StrictBool # Wheter or not to plot the intensity of the field near the focus
        focus_field_intensity_figure_name: StrictStr = 'Intensity at the focus'
        
        plot_focus_field_amplitude: StrictBool # Wheter or not to plot the amplitude and phase of the field near the focus
        focus_field_amplitude_figure_name: StrictStr = 'Amplitude at the focus'
        

    def __init__(self, mask_type: MaskType) -> None:
        _objective_field_calculator, _focus_field_calculator, self.acount_for_pixel_width = self._STRATEGY_MAPPER[mask_type]
        self._objective_field_calculator = _objective_field_calculator()
        self._focus_field_calculator = _focus_field_calculator()
        
        self.plot_objective_field_function = plot_objective_field
    
    def _handle_propagated_field_calculation(self, basic_parameters, objective_field_parameters):
        """Not implemented, since this was an unlikely use for the program"""
        raise NotImplementedError
        E_rho,Ex,Ey = self._objective_field_calculator()
        if basic_parameters.plot_incident_field == True:
            self.plot_objective_field_function(Ex,Ey, objective_field_parameters)
        fields = self._focus_field_calculator(E_rho)
    
    def calculate_2D_fields(
            self, 
            basic_parameters: BasicParameters, 
            objective_field_parameters: FreePropagationCalculator.ObjectiveFieldParameters, 
            focus_field_parameters: FocusFieldCalculator.FocusFieldParameters,
            **kwargs
        ) -> FocusFieldCalculator.FieldAtFocus:
        """Main function that calculates the field along a plane. Orchestrates the steps of the calculation, like handling the conversion between different units of measurement (mm to nm, degrees to radians, etc...).

        Args:
            :basic_parameters (BasicParameters): Basic configuration of the simulation, that defines if the amplitude is to be plot and with what name
            :objective_field_parameters: Currently unused since the calculation with propagation is not implemented
            :focus_field_parameters (FocusFieldCalculator.FocusFieldParameters): Parameters of the simulation

        Returns:
            FocusFieldCalculator.FieldAtFocus: Field near the focus on a plane. Contains the 3 components and the intensity
        """
        logger.info("PyFocus: Performing a simulation along the XY and XZ planes")
        if basic_parameters.propagate_incident_field:
            objective_field_parameters.transform_input_parameter_units()
            return self._handle_propagated_field_calculation(basic_parameters, objective_field_parameters)        
        
        if focus_field_parameters.interface_parameters is None:
            focus_field_parameters.field_parameters.wavelength /= focus_field_parameters.n
        
        focus_field_parameters.transform_input_parameter_units()
        focus_field = self._focus_field_calculator.calculate(focus_field_parameters,**kwargs)
        
        if basic_parameters.plot_focus_field_intensity == True:
            plot_params=PlotParameters(name=basic_parameters.focus_field_intensity_figure_name) #TODO añadir el size
            plot_intensity_at_focus(focus_field, focus_field_parameters, params=plot_params, acount_for_pixel_width=self.acount_for_pixel_width)
        
        if basic_parameters.plot_focus_field_amplitude == True:
            plot_params=PlotParameters(name=basic_parameters.focus_field_amplitude_figure_name)
            plot_amplitude_and_phase_at_focus(focus_field, focus_field_parameters, params=plot_params, acount_for_pixel_width=self.acount_for_pixel_width)
        
        plt.show()
        return focus_field
    
    def calculate_3D_fields(
            self, 
            basic_parameters: BasicParameters, 
            objective_field_parameters: FreePropagationCalculator.ObjectiveFieldParameters, 
            focus_field_parameters: FocusFieldCalculator.FocusFieldParameters,
            progress_callback: callable = None,
            **kwargs
        ) -> FocusFieldCalculator.FieldAtFocus3D:
        """Main function that calculates the field along a 3D space by calculating it as slices of 2D XY planes. Orchestrates the steps of the calculation, like handling the conversion between different units of measurement (mm to nm, degrees to radians, etc...).

        Args:
            :basic_parameters (BasicParameters): Basic configuration of the simulation, that defines if the amplitude is to be plot and with what name
            :objective_field_parameters: Currently unused since the calculation with propagation is not implemented
            :focus_field_parameters (FocusFieldCalculator.FocusFieldParameters): Parameters of the simulation

        Returns:
            FocusFieldCalculator.FieldAtFocus: Field near the focus on a plane. Contains the 3 components of the field and the intensity
        """
        logger.info("PyFocus: Performing a simulation in a 3D space")
                
        focus_field_parameters.transform_input_parameter_units()
        fields = self._focus_field_calculator.calculate_3D_field(focus_field_parameters, progress_callback=progress_callback,**kwargs)
        fields.calculate_intensity()
        
        # logger.debug("PyFocus: Obtained fields:")
        # shape = np.shape(fields.Intensity)
        # logger.debug(f"Intensity at the center: {fields.Intensity[shape[0]//2, shape[1]//2, shape[2]//2]}")
        # logger.debug(f"{np.mean(fields.Ex[shape[0]//2,:,:])=}")
        # logger.debug(f"{np.mean(fields.Ey[shape[0]//2,:,:])=}")
        # logger.debug(f"{np.mean(fields.Ez[shape[0]//2,:,:])=}")
        # logger.debug("")
        
        
        if basic_parameters.plot_focus_field_intensity == True:
            plot_params=PlotParameters(name=basic_parameters.focus_field_intensity_figure_name) #TODO añadir el size
            plot_3D_intensity_at_focus(fields, focus_field_parameters, params=plot_params, acount_for_pixel_width=self.acount_for_pixel_width)
        
        if basic_parameters.plot_focus_field_amplitude == True:
            plot_params=PlotParameters(name=basic_parameters.focus_field_amplitude_figure_name)
            plot_3D_amplitude_and_phase_at_focus(fields, focus_field_parameters, params=plot_params, acount_for_pixel_width=self.acount_for_pixel_width)
        
        # plt.show()
        return fields
