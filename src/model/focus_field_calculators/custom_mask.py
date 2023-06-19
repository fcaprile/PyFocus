"""
Functions for the simulation of the field obtained by focuisng a gaussian beam
"""
from typing import Callable
from ...custom_dataclasses.custom_mask import PlotPlanes
from ...model.focus_field_calculators.base import FocusFieldCalculator
from ...log_config import logger
import numpy as np
from tqdm import tqdm
from ...equations.gaussian_profile import gaussian_rho
from ...plot_functions.plot_objective_field import plot_in_cartesian
from ...equations.tmm_core import coh_tmm

class CustomMaskFocusFieldCalculator(FocusFieldCalculator):
    """Class that performs the calculation of the field near the focus for a custom mask"""
    
    # Text that is displayed while performing the numerical calculation:
    XY_FIELD_DESCRIPTION = 'Custom Mask calculation along XY plane'
    XZ_FIELD_DESCRIPTION = 'Custom Mask calculation along XZ plane'
    
    def calculate_3D_field(self, focus_field_parameters: FocusFieldCalculator.FocusFieldParameters, mask_function: callable, progress_callback: callable = None) -> FocusFieldCalculator.FieldAtFocus3D:
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
        
        custom_field_function=lambda rho, phi,w0,f,k: mask_function(rho, phi,w0,f,k)
        ex_lens,ey_lens=self._generate_rotated_incident_field(custom_field_function, focus_field_parameters)
        logger.debug(f"{focus_field_parameters.z_step_count=}, {focus_field_parameters.r_step_count=}")
        Ex,Ey,Ez = [np.zeros((focus_field_parameters.z_step_count, focus_field_parameters.r_step_count, focus_field_parameters.r_step_count),dtype=complex) for _ in range(3)]
        axial_positions = focus_field_parameters.z_steps * (np.arange(focus_field_parameters.z_step_count) - focus_field_parameters.z_step_count // 2)
        # axial_positions = np.linspace(-focus_field_parameters.z_range,focus_field_parameters.z_range,focus_field_parameters.z_step_count)
        if progress_callback:
            progress_callback("Simulation with vectorial propagation model has started")
        for index in tqdm(range(len(axial_positions)),desc='Calculating the field at each XY plane'):
            axial_distance = axial_positions[index]
            if progress_callback:
                progress_callback(f"Calculating the field at z={axial_distance}nm. Do not modify the simulation parameters.")
            focus_field_parameters.z = axial_distance
            # logger.info(f"Calculating the field at z={axial_distance}nm")
            Ex[index,:,:], Ey[index,:,:], Ez[index,:,:] = self._calculate_field_along_XY_plane(ex_lens,ey_lens, focus_field_parameters, verbose=False)
        
        return self.FieldAtFocus3D(Ex,Ey,Ez)
    
    def calculate(self, focus_field_parameters: FocusFieldCalculator.FocusFieldParameters, mask_function: callable) -> FocusFieldCalculator.FieldAtFocus:
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
                
        custom_field_function=lambda rho, phi,w0,f,k: (gaussian_rho(w0))(rho)* mask_function(rho, phi,w0,f,k)
        ex_lens,ey_lens=self._generate_rotated_incident_field(custom_field_function, focus_field_parameters)
        Ex,Ey,Ez = self._calculate_field_along_XZ_plane(ex_lens,ey_lens, focus_field_parameters, verbose=True)
        Ex2,Ey2,Ez2 = self._calculate_field_along_XY_plane(ex_lens,ey_lens, focus_field_parameters, verbose=True)
        
        return self.FieldAtFocus(Ex,Ey,Ez,Ex2,Ey2,Ez2)
    
    def _calculate_field_along_XY_plane(self, ex_lens,ey_lens, focus_field_parameters: FocusFieldCalculator.FocusFieldParameters, verbose: bool=True):
        """Intermediary function to calculate the field along the XY plane, separated for modularity of the code and simplicity"""
        return self._calculate_field_along_a_plane(
            ex_lens,
            ey_lens, 
            focus_field_parameters.x_range/2, 
            focus_field_parameters.x_range/2, 
            focus_field_parameters.x0, 
            focus_field_parameters.y0, 
            focus_field_parameters.r_step_count,
            focus_field_parameters.r_step_count,
            focus_field_parameters, 
            description=self.XY_FIELD_DESCRIPTION,
            plane_to_plot=PlotPlanes.XY,
            verbose=verbose)

    def _calculate_field_along_XZ_plane(self, ex_lens,ey_lens, focus_field_parameters: FocusFieldCalculator.FocusFieldParameters, verbose: bool=True):
        """Intermediary function to calculate the field along the XZ plane, separated for modularity of the code and simplicity"""
        return self._calculate_field_along_a_plane(
            ex_lens,
            ey_lens, 
            focus_field_parameters.x_range/2, 
            focus_field_parameters.z_range/2, 
            focus_field_parameters.x0, 
            focus_field_parameters.z, 
            focus_field_parameters.r_step_count,
            focus_field_parameters.z_step_count,
            focus_field_parameters, 
            description=self.XZ_FIELD_DESCRIPTION,
            plane_to_plot=PlotPlanes.XZ,
            verbose=verbose)
    
    def _calculate_field_along_a_plane(self, ex_lens, ey_lens, horizontal_max, vertical_max, horizontal_0, vertical_0, horizonal_steps, vertical_steps, focus_field_parameters: FocusFieldCalculator.FocusFieldParameters, description: str, plane_to_plot: PlotPlanes, verbose: bool=True):
        """Parent function that delegates the calculation to the corresponding functions.
        Returns:
            The 3 matrixes of each component of the field 
        """
                
        horizontal_values, vertical_values = self._calculate_cartesian_coordinates(horizontal_max, vertical_max, horizontal_0, vertical_0, horizonal_steps, vertical_steps)
        # logger.debug("Calculating the field algong a plane...")
        # logger.debug(f"{focus_field_parameters.z=}")
        # logger.debug(f"{focus_field_parameters.interface_parameters=}")
        # logger.debug(f"{horizontal_max=}")
        # logger.debug(f"{vertical_max=}")
        # logger.debug(f"{horizontal_0=}")
        # logger.debug(f"{vertical_0=}")
        # logger.debug(f"{horizonal_steps=}")
        # logger.debug(f"{vertical_steps=}")
        if focus_field_parameters.interface_parameters is None:
            factors = self._calculate_factors_without_interface(ex_lens, ey_lens, focus_field_parameters)
            ex, ey, ez = self._integrate_without_interface(horizontal_values, vertical_values, factors, focus_field_parameters, description, plane_to_plot, verbose=verbose)
        else:
            factors = self._calculate_factors_with_interface(ex_lens, ey_lens, focus_field_parameters)
            ex, ey, ez = self._integrate_with_interface(horizontal_values, vertical_values, factors, focus_field_parameters, description, plane_to_plot, verbose=verbose)
                
        return ex,ey,ez
    
    def _calculate_2D_trapezoidal_method_weight(self, alpha, divisions_theta, divisions_phi):
        """Calculates the numerical weights used in the numerical integration
        """
        h_theta=alpha/divisions_theta
        h_phi=2*np.pi/divisions_phi
        weight_trapezoid_rho=np.zeros(divisions_theta)+h_theta
        weight_trapezoid_rho[0]=h_theta/2
        weight_trapezoid_rho[-1]=h_theta/2
        weight_trapezoid_phi=np.zeros(divisions_phi)+h_phi
        weight_trapezoid_phi[0]=h_phi/2
        weight_trapezoid_phi[-1]=h_phi/2
        return weight_trapezoid_rho*np.vstack(weight_trapezoid_phi)#represents the area of each trapezoid for each position in phi,theta

    def _initialize_fields(self, x_size: int, y_size: int) -> list[list]:
        """Returns 3 empty arrays that are later filled with the calculated values of the field
        """
        return [np.zeros((x_size, y_size),dtype=complex) for _ in range(3)]
    
    def _calculate_cartesian_coordinates(self, horizontal_max, vertical_max, horizontal_0, vertical_0, horizonal_steps, vertical_steps):
        """Calculates the cartesian coordinates used later for the numerical integration. It is separated from the main calculation for modularity of the code and simplicity
        """
        horizontal_values = np.linspace(-horizontal_max+horizontal_0,horizontal_max+horizontal_0,horizonal_steps)
        vertical_values = np.linspace(vertical_max+vertical_0,-vertical_max+vertical_0,vertical_steps)
        return horizontal_values, vertical_values
    
    def _calculate_polar_coordinates(self, focus_field_parameters: FocusFieldCalculator.FocusFieldParameters):
        """Calculates the polar coordinates used later for the numerical integration. It is separated from the main calculation for modularity of the code and simplicity
        """
        theta_values=np.linspace(0,focus_field_parameters.alpha,focus_field_parameters.custom_mask_parameters.divisions_theta)    #divisions of theta in which the trapezoidal 2D integration is done
        phi_values=np.linspace(0,2*np.pi,focus_field_parameters.custom_mask_parameters.divisions_phi)      #divisions of phi in which the trapezoidal 2D integration is done
        return np.meshgrid(theta_values,phi_values)       #turn the divisions into a matrix in order to apply the function more easily
    
    def _calculate_factors_without_interface(self, ex_lens, ey_lens, focus_field_parameters: FocusFieldCalculator.FocusFieldParameters):
        """Calculates the numerical values used later for the numerical integration. It is separated from the main calculation for modularity of the code and simplicity
        """
        weight_trapezoid = self._calculate_2D_trapezoidal_method_weight(focus_field_parameters.alpha, focus_field_parameters.custom_mask_parameters.divisions_theta, focus_field_parameters.custom_mask_parameters.divisions_phi)
        kz=focus_field_parameters.z*2*np.pi/focus_field_parameters.field_parameters.wavelength
        theta, phi = self._calculate_polar_coordinates(focus_field_parameters)
        #now begins the integration, in order to save computing time i do the trigonometric functions separatedly and save the value into another variable
        cos_theta=np.cos(theta)
        cos_theta_sqrt=cos_theta**0.5
        sin_theta=np.sin(theta)
        cos_phi=np.cos(phi)
        sin_phi=np.sin(phi)
        sin_phi_square=sin_phi**2
        
        prefactor_general=cos_theta_sqrt*sin_theta
        prefactor_x=prefactor_general*(cos_theta+(1-cos_theta)*sin_phi_square)
        prefactor_y=prefactor_general*(1-cos_theta)*cos_phi*sin_phi
        prefactor_z=prefactor_general*(-sin_theta*cos_phi)
        
        Axx=prefactor_x*ex_lens*weight_trapezoid
        Axy=prefactor_y*ex_lens*weight_trapezoid
        Axz=prefactor_z*ex_lens*weight_trapezoid

        Ayx=prefactor_y*ey_lens*weight_trapezoid
        Ayy=-prefactor_x*ey_lens*weight_trapezoid
        Ayz=prefactor_z*ey_lens*weight_trapezoid

        cos_theta_kz=cos_theta*kz
        
        # logger.debug("Calculating intermediary factors:")
        # logger.debug("")
        # logger.debug("Input:")
        # parameters = (ex_lens,ey_lens,focus_field_parameters.alpha,focus_field_parameters.h,focus_field_parameters.field_parameters.wavelength,focus_field_parameters.z,focus_field_parameters.x_steps,focus_field_parameters.custom_mask_parameters.divisions_theta,focus_field_parameters.custom_mask_parameters.divisions_phi,focus_field_parameters.x_range)
        # parameter_names = ("ex_lens","ey_lens","alpha","h","wavelength","zp0","resolution_x","divisions_theta","divisions_phi","x_range")
        # for parameter, name in zip(parameters, parameter_names):
        #     if isinstance(parameter, float) or isinstance(parameter, bool) or isinstance(parameter, int) or isinstance(parameter, np.int64) or isinstance(parameter, np.int32) or len(parameter) < 5:
        #         logger.debug(f"{name}: {parameter}")
        #     else:
        #         logger.debug(f"np.mean({name}): {np.mean(parameter)}")
        # logger.debug("")            
        # logger.debug("Output:")
        # logger.debug(f"{np.mean(Axx)=}")
        # logger.debug(f"{np.mean(Axy)=}")
        # logger.debug(f"{np.mean(Axz)=}")
        # logger.debug(f"{np.mean(Ayx)=}")
        # logger.debug(f"{np.mean(Ayy)=}")
        # logger.debug(f"{np.mean(Ayz)=}")
        # logger.debug(f"{np.mean(cos_theta_kz)=}")
        # logger.debug(f"{np.mean(cos_theta)=}")
        # logger.debug(f"{np.mean(sin_theta)=}")
        # logger.debug(f"{np.mean(phi)=}")
        
        return Axx, Axy, Axz, Ayx, Ayy, Ayz, cos_theta_kz, cos_theta, sin_theta, phi
    
    def _calculate_factors_with_interface(self, ex_lens, ey_lens, focus_field_parameters: FocusFieldCalculator.FocusFieldParameters):
        """Calculates the numerical values used later for the numerical integration. It is separated from the main calculation for modularity of the code and simplicity
        """
        weight_trapezoid = self._calculate_2D_trapezoidal_method_weight(focus_field_parameters.alpha, focus_field_parameters.custom_mask_parameters.divisions_theta, focus_field_parameters.custom_mask_parameters.divisions_phi)
        theta, phi = self._calculate_polar_coordinates(focus_field_parameters)
        #now begins the integration, in order to save computing time i do the trigonometric functions separatedly and save the value into another variable
        cos_theta=np.cos(theta)
        cos_theta_sqrt=cos_theta**0.5
        sin_theta=np.sin(theta)
        cos_phi=np.cos(phi)
        sin_phi=np.sin(phi)
        sin_phi_square=sin_phi**2
        
        n1 = focus_field_parameters.interface_parameters.ns[0]
        n2 = focus_field_parameters.interface_parameters.ns[-1]
        k1=2*np.pi/focus_field_parameters.field_parameters.wavelength*n1
        k2=k1*n2/n1

        prefactor_general=cos_theta_sqrt*sin_theta*k1
        prefactor_x=prefactor_general*(cos_theta+(1-cos_theta)*sin_phi_square)
        prefactor_y=prefactor_general*(1-cos_theta)*cos_phi*sin_phi
        prefactor_z=prefactor_general*(-sin_theta*cos_phi)
        
        Axx=-prefactor_x*ex_lens*weight_trapezoid
        Axy=-prefactor_y*ex_lens*weight_trapezoid
        Axz=-prefactor_z*ex_lens*weight_trapezoid

        Ayx=-prefactor_y*ey_lens*weight_trapezoid
        Ayy=prefactor_x*ey_lens*weight_trapezoid
        Ayz=-prefactor_z*ey_lens*weight_trapezoid
                
        #Calculus of the refraction and transmition coeficients
        rs_i_theta=np.zeros((focus_field_parameters.custom_mask_parameters.divisions_phi,focus_field_parameters.custom_mask_parameters.divisions_theta),dtype='complex')
        rp_i_theta=np.copy(rs_i_theta)
        ts_t_theta=np.copy(rs_i_theta)
        tp_t_theta=np.copy(rs_i_theta)
        theta_values=np.linspace(0,focus_field_parameters.alpha,focus_field_parameters.custom_mask_parameters.divisions_theta) 
        for i, theta_val in enumerate(theta_values):
            tmm_p=coh_tmm('p', focus_field_parameters.interface_parameters.ns, focus_field_parameters.interface_parameters.ds, theta_val, focus_field_parameters.field_parameters.wavelength)
            tmm_s=coh_tmm('s', focus_field_parameters.interface_parameters.ns, focus_field_parameters.interface_parameters.ds, theta_val, focus_field_parameters.field_parameters.wavelength)
            rs_i_theta[:,i]=tmm_s['r']
            rp_i_theta[:,i]=tmm_p['r']
            ts_t_theta[:,i]=tmm_s['t']
            tp_t_theta[:,i]=tmm_p['t']
        
        #For integration of the reflected and transmited fields (Er and Et):
        prefactor_x_r=-prefactor_general*(-rs_i_theta*sin_phi_square+rp_i_theta*cos_phi**2*cos_theta)
        prefactor_y_r=-prefactor_general*(rs_i_theta+rp_i_theta*cos_theta)*cos_phi*sin_phi
        prefactor_z_r=prefactor_general*rp_i_theta*(-sin_theta*cos_phi)
        
        phase_z_r=np.exp(2*1j*k1*np.cos(theta)*focus_field_parameters.interface_parameters.axial_position)
        
        Axx_r=-prefactor_x_r*ex_lens*weight_trapezoid
        Axy_r=prefactor_y_r*ex_lens*weight_trapezoid
        Axz_r=prefactor_z_r*ex_lens*weight_trapezoid

        Ayx_r=phase_z_r*prefactor_y_r*ey_lens*weight_trapezoid
        Ayy_r=phase_z_r*prefactor_x_r*ey_lens*weight_trapezoid
        Ayz_r=phase_z_r*prefactor_z_r*ey_lens*weight_trapezoid
    
        #switching to complex angles in order to compute the transmited complex angles:
        theta_values_complex=np.linspace(0,focus_field_parameters.alpha,focus_field_parameters.custom_mask_parameters.divisions_theta,dtype='complex')    
        phi_values_complex=np.linspace(0,2*np.pi,focus_field_parameters.custom_mask_parameters.divisions_phi,dtype='complex') 
        theta_complex,phi_complex=np.meshgrid(theta_values_complex,phi_values_complex)

        sin_theta_complex=np.sin(theta_complex)

        cos_theta_t=(1-(n1/n2*sin_theta_complex)**2)**0.5
        sin_theta_t=n1/n2*sin_theta #snell
        prefactor_general_t=(cos_theta)**0.5*sin_theta*k1

        prefactor_x_t=prefactor_general_t*(ts_t_theta*sin_phi_square+tp_t_theta*cos_phi**2*cos_theta_t)
        prefactor_y_t=prefactor_general_t*(-ts_t_theta+tp_t_theta*cos_theta_t)*cos_phi*sin_phi
        prefactor_z_t=prefactor_general_t*tp_t_theta*sin_theta_t*cos_phi
        
        phase_z_t=np.exp(1j*focus_field_parameters.interface_parameters.axial_position*(k2*cos_theta_t+k1*cos_theta))
        
        Axx_t=-phase_z_t*prefactor_x_t*ex_lens*weight_trapezoid
        Axy_t=phase_z_t*prefactor_y_t*ex_lens*weight_trapezoid
        Axz_t=phase_z_t*prefactor_z_t*ex_lens*weight_trapezoid

        Ayx_t=phase_z_t*prefactor_y_t*ey_lens*weight_trapezoid
        Ayy_t=phase_z_t*prefactor_x_t*ey_lens*weight_trapezoid
        Ayz_t=phase_z_t*prefactor_z_t*ey_lens*weight_trapezoid
        
        # logger.debug("Calculating intermediary factors:")
        # logger.debug("")
        # logger.debug("Input:")
        # parameters = (n1,n2,focus_field_parameters.interface_parameters.ns,focus_field_parameters.interface_parameters.ds,ex_lens,ey_lens,focus_field_parameters.alpha,focus_field_parameters.h,focus_field_parameters.field_parameters.wavelength,focus_field_parameters.interface_parameters.axial_position,focus_field_parameters.z,focus_field_parameters.x_steps,focus_field_parameters.custom_mask_parameters.divisions_theta,focus_field_parameters.custom_mask_parameters.divisions_phi,focus_field_parameters.x_range)
        # parameter_names = ("n1","n2","n_list","d_list","ex_lens","ey_lens","alpha","h","wavelength","z_int","zp0","resolution_x","divisions_theta","divisions_phi","x_range")
        # for parameter, name in zip(parameters, parameter_names):
        #     if isinstance(parameter, float) or isinstance(parameter, bool) or isinstance(parameter, int) or isinstance(parameter, np.int64) or isinstance(parameter, np.int32) or len(parameter) < 5:
        #         logger.debug(f"{name}: {parameter}")
        #     else:
        #         logger.debug(f"np.mean({name}): {np.mean(parameter)}")
        # logger.debug("")            
        # logger.debug("Output:")
        # logger.debug(f"{np.mean(prefactor_x_r)=}")
        # logger.debug(f"{np.mean(prefactor_y_r)=}")
        # logger.debug(f"{np.mean(prefactor_z_r)=}")
        # logger.debug(f"{np.mean(phase_z_r)=}")
        # logger.debug(f"{np.mean(Axx_r)=}")
        # logger.debug(f"{np.mean(Axy_r)=}")
        # logger.debug(f"{np.mean(Axz_r)=}")
        # logger.debug(f"{np.mean(Ayx_r)=}")
        # logger.debug(f"{np.mean(Ayy_r)=}")
        # logger.debug(f"{np.mean(Ayz_r)=}")
        # logger.debug(f"{np.mean(theta_values_complex)=}")
        # logger.debug(f"{np.mean(phi_values_complex)=}")
        # logger.debug(f"{np.mean(Axx_t)=}")
        # logger.debug(f"{np.mean(Axy_t)=}")
        # logger.debug(f"{np.mean(Axz_t)=}")
        # logger.debug(f"{np.mean(Ayx_t)=}")
        # logger.debug(f"{np.mean(Ayy_t)=}")
        # logger.debug(f"{np.mean(Ayz_t)=}")
        
        return Axx, Axy, Axz, Ayx, Ayy, Ayz, Axx_r, Axy_r, Axz_r, Ayx_r, Ayy_r, Ayz_r, Axx_t, Axy_t, Axz_t, Ayx_t, Ayy_t, Ayz_t, cos_theta, cos_theta_t, sin_theta, phi, k1, k2
    
    def _integrate_without_interface(self, horizontal_values: list[int], vertical_values: list[int], factors: list[any], focus_field_parameters: FocusFieldCalculator.FocusFieldParameters, description: str, plane_to_plot: PlotPlanes, verbose: bool=True):
        """Performs the numerical integration for the simmulation without an interface (a tipical simulation)

        Args:
            horizontal_values (list[int]): Values for the horizontal axis of the simulation
            vertical_values (list[int]): Values for the vertical axis of the simulation
            factors (list[any]): Numerical values calculated previously that are used in the calculation
            focus_field_parameters (FocusFieldCalculator.FocusFieldParameters): Parameters of the simulation
            description (str): Text that is displayed while the integral is being performed
            plane_to_plot (PlotPlanes): Plane along which to integrate (XZ, YZ or XY)

        Raises:
            NotImplementedError: In case wrong parameters are passed for the plane to plot

        Returns:
            The 3 matrixes of each component of the field 
        """
        Axx, Axy, Axz, Ayx, Ayy, Ayz, cos_theta_kz, cos_theta, sin_theta, phi = factors
        ex, ey, ez = self._initialize_fields(len(vertical_values), len(horizontal_values))
        def _integrate_XZ_or_YZ(j):
            zp0=vertical_values[j]
            for i,x in enumerate(horizontal_values):
                rhop=np.abs(x)
                phip=np.arctan2(0,x)
                kr=rhop*2*np.pi/focus_field_parameters.field_parameters.wavelength
                kz=zp0*2*np.pi/focus_field_parameters.field_parameters.wavelength
                sin_theta_kr=sin_theta*kr
                cos_theta_kz=cos_theta*kz
                phase_inc_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip) + cos_theta_kz))#phase for the X incident component
                phase_inc_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip) + cos_theta_kz))#phase for the Y incident component
                #now, the integration is made as the sum of the value of the integrand in each position of phi,theta:
                ex[j,i]=np.sum(Axx*phase_inc_x)+np.sum(Ayx*phase_inc_y)
                ey[j,i]=np.sum(Axy*phase_inc_x)+np.sum(Ayy*phase_inc_y)
                ez[j,i]=np.sum(Axz*phase_inc_x)+np.sum(Ayz*phase_inc_y)
        
        def _integrate_XY(i):
            x=horizontal_values[i]
            for j,y in enumerate(vertical_values):
                rhop=(x**2+y**2)**0.5
                phip=np.arctan2(y,x)
                kr=rhop*2*np.pi/focus_field_parameters.field_parameters.wavelength
                sin_theta_kr=sin_theta*kr
                phase_inc_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip) + cos_theta_kz))#phase for the X incident component
                phase_inc_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip) + cos_theta_kz))#phase for the Y incident component
                #now, the integration is made as the sum of the value of the integrand in each position of phi,theta:
                ex[j,i]=np.sum(Axx*phase_inc_x)+np.sum(Ayx*phase_inc_y)
                ey[j,i]=np.sum(Axy*phase_inc_x)+np.sum(Ayy*phase_inc_y)
                ez[j,i]=np.sum(Axz*phase_inc_x)+np.sum(Ayz*phase_inc_y)
        
        #Para mejorar la velocidad del calculo:
        #TODO prearmar los valores de rhop, phip, kr, sin_theta_kr
        #rhop = (x_values**2+y_values**2)**0.5
        #phip=np.arctan2(y_values,x_values)
        #now for each position in which i calculate the field i do the integration
        if plane_to_plot == PlotPlanes.XZ or plane_to_plot == PlotPlanes.YZ:
            if verbose == True:
                for j in tqdm(range(focus_field_parameters.z_step_count),desc=description):
                    _integrate_XZ_or_YZ(j)
            else:
                for j in range(focus_field_parameters.z_step_count):
                    _integrate_XZ_or_YZ(j)
        elif plane_to_plot == PlotPlanes.XY:
            if verbose == True:
                for i in tqdm(range(focus_field_parameters.r_step_count),desc=description):
                    _integrate_XY(i)
            else:
                for i in range(focus_field_parameters.r_step_count):
                    _integrate_XY(i)
        else:
            raise NotImplementedError
        
        ex*=-1j*focus_field_parameters.f/focus_field_parameters.field_parameters.wavelength
        ey*=1j*focus_field_parameters.f/focus_field_parameters.field_parameters.wavelength
        ez*=1j*focus_field_parameters.f/focus_field_parameters.field_parameters.wavelength
        
        return ex, ey, ez

    def _integrate_with_interface(self, horizontal_values: list[int], vertical_values: list[int], factors: list[any], focus_field_parameters: FocusFieldCalculator.FocusFieldParameters, description: str, plane_to_plot: PlotPlanes, verbose: bool=True):
        """Performs the numerical integration for the simmulation with an interface

        Args:
            horizontal_values (list[int]): Values for the horizontal axis of the simulation
            vertical_values (list[int]): Values for the vertical axis of the simulation
            factors (list[any]): Numerical values calculated previously that are used in the calculation
            focus_field_parameters (FocusFieldCalculator.FocusFieldParameters): Parameters of the simulation
            description (str): Text that is displayed while the integral is being performed
            plane_to_plot (PlotPlanes): Plane along which to integrate (XZ, YZ or XY)

        Raises:
            NotImplementedError: In case wrong parameters are passed for the plane to plot

        Returns:
            The 3 matrixes of each component of the field 
        """
        Axx, Axy, Axz, Ayx, Ayy, Ayz, Axx_r, Axy_r, Axz_r, Ayx_r, Ayy_r, Ayz_r, Axx_t, Axy_t, Axz_t, Ayx_t, Ayy_t, Ayz_t, cos_theta, cos_theta_t, sin_theta, phi, k1, k2 = factors
        ex, ey, ez = self._initialize_fields(len(vertical_values), len(horizontal_values))

        def _integrate_XZ_or_YZ(j):
            zp0=vertical_values[j]
            for i,x in enumerate(horizontal_values):
                rhop=np.abs(x)
                phip=np.arctan2(0,x)
                kr=rhop*k1
                sin_theta_kr=sin_theta*kr       #because of snell's law, this factor will be the same for the reflected and transmited fields
                
                phase_rho_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip)))
                phase_rho_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip)))
                
                if focus_field_parameters.interface_parameters.axial_position<=zp0:
                    kz_t=zp0*k2
                    phase_kz_t=np.exp(1j*cos_theta_t*kz_t)
                    phase_inc_x=phase_rho_x*phase_kz_t      #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz_t      #phase for the Y incident component of the transmited field
                    ex[j,i]=np.sum(Axx_t*phase_inc_x)+np.sum(Ayx_t*phase_inc_y)
                    ey[j,i]=np.sum(Axy_t*phase_inc_x)+np.sum(Ayy_t*phase_inc_y)
                    ez[j,i]=np.sum(Axz_t*phase_inc_x)+np.sum(Ayz_t*phase_inc_y)
                else:
                    kz=zp0*k1
                    phase_kz=np.exp(1j*cos_theta*kz)
                    phase_kz_r=np.exp(-1j*cos_theta*kz)
                    phase_inc_x=phase_rho_x*phase_kz        #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz        #phase for the Y incident component of the transmited field
                    phase_inc_x_r=phase_rho_x*phase_kz_r    #phase for the X incident component of the reflected field
                    phase_inc_y_r=phase_rho_y*phase_kz_r    #phase for the Y incident component of the reflected field
                    ex[j,i]=np.sum(Axx*phase_inc_x+Axx_r*phase_inc_x_r)+np.sum(Ayx*phase_inc_y+Ayx_r*phase_inc_y_r)
                    ey[j,i]=np.sum(Axy*phase_inc_x+Axy_r*phase_inc_x_r)+np.sum(Ayy*phase_inc_y+Ayy_r*phase_inc_y_r)
                    ez[j,i]=np.sum(Axz*phase_inc_x+Axz_r*phase_inc_x_r)+np.sum(Ayz*phase_inc_y+Ayz_r*phase_inc_y_r)

        def _integrate_XY(i):
            x=horizontal_values[i]
            for j,y in enumerate(vertical_values):
                rhop=(x**2+y**2)**0.5
                phip=np.arctan2(y,x)
                kr=rhop*k1
                sin_theta_kr=sin_theta*kr
                
                phase_rho_x=np.exp(1j*(sin_theta_kr*np.cos(phi - phip)))
                phase_rho_y=np.exp(1j*(-sin_theta_kr*np.sin(phi - phip)))
                
                if focus_field_parameters.interface_parameters.axial_position<=zp0:
                    phase_inc_x=phase_rho_x*phase_kz_t      #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz_t      #phase for the Y incident component of the transmited field
                    ex[j,i]=np.sum(Axx_t*phase_inc_x)+np.sum(Ayx_t*phase_inc_y)
                    ey[j,i]=np.sum(Axy_t*phase_inc_x)+np.sum(Ayy_t*phase_inc_y)
                    ez[j,i]=np.sum(Axz_t*phase_inc_x)+np.sum(Ayz_t*phase_inc_y)
                else:
                    phase_inc_x=phase_rho_x*phase_kz        #phase for the X incident component of the transmited field
                    phase_inc_y=phase_rho_y*phase_kz        #phase for the Y incident component of the transmited field
                    phase_inc_x_r=phase_rho_x*phase_kz_r    #phase for the X incident component of the reflected field
                    phase_inc_y_r=phase_rho_y*phase_kz_r    #phase for the Y incident component of the reflected field
                    ex[j,i]=np.sum(Axx*phase_inc_x+Axx_r*phase_inc_x_r)+np.sum(Ayx*phase_inc_y+Ayx_r*phase_inc_y_r)
                    ey[j,i]=np.sum(Axy*phase_inc_x+Axy_r*phase_inc_x_r)+np.sum(Ayy*phase_inc_y+Ayy_r*phase_inc_y_r)
                    ez[j,i]=np.sum(Axz*phase_inc_x+Axz_r*phase_inc_x_r)+np.sum(Ayz*phase_inc_y+Ayz_r*phase_inc_y_r)

        if plane_to_plot == PlotPlanes.XZ or plane_to_plot == PlotPlanes.YZ:
            if verbose == True:
                for j in tqdm(range(focus_field_parameters.z_step_count),desc=description):
                    _integrate_XZ_or_YZ(j)
            else:
                for j in range(focus_field_parameters.z_step_count):
                    _integrate_XZ_or_YZ(j)
        elif plane_to_plot == PlotPlanes.XY:
            zp0 = focus_field_parameters.z
            kz=zp0*k1
            kz_r=zp0*k1
            kz_t=zp0*k2
            phase_kz=np.exp(1j*cos_theta*kz)
            phase_kz_r=np.exp(-1j*cos_theta*kz_r)
            phase_kz_t=np.exp(1j*cos_theta_t*kz_t)
            if verbose == True:
                for i in tqdm(range(focus_field_parameters.r_step_count),desc=description):
                    _integrate_XY(i)
            else:
                for i in range(focus_field_parameters.r_step_count):
                    _integrate_XY(i)
        else:
            raise NotImplementedError
                
        ex*=-1j*focus_field_parameters.f/2/np.pi*np.exp(-1j*k1*focus_field_parameters.f)
        ey*=1j*focus_field_parameters.f/2/np.pi*np.exp(-1j*k1*focus_field_parameters.f)
        ez*=1j*focus_field_parameters.f/2/np.pi*np.exp(-1j*k1*focus_field_parameters.f)
        return ex, ey, ez

    def _generate_rotated_incident_field(self, maskfunction: Callable, focus_field_parameters: FocusFieldCalculator.FocusFieldParameters):
        """Generates the incident field for the future numerical integration. The term "rotated" makes reference to the fact that in the integral, the Y component is evaluated on phi-pi/2

        Args:
            :maskfunction (Callable): Analytical function that defines the phase mask, must be a function of the 5 internal variables: rho, phi, w0, f and k, with:
                
                rho: Radial coordinate from 0 to the aperture radius of the objective.
                
                phi: Azimutal coordinate from 0 to 2pi.
                
                w0: Radius of the incident gaussian beam.
                
                f: Focal distane of the objective lens (mm)
                
                k: Wavenumber in the objective lens medium (mm)
                
                The real part defines the amplitude of the incident field
            
            focus_field_parameters (FocusFieldCalculator.FocusFieldParameters): Parameters descibing the simulation

        Returns:
            2 matrixes that define the field that is inciding on the lens. 
            These arrays have the value of the amplitude of the incident field for each value of theta and phi: ex_lens[phi_position,theta_position]
            for phi_position a value in np.linspace(0,2*np.pi,divisions_phi) and theta_position a value in np.linspace(0,alpha,divisions_theta) 
        """
        k=2*np.pi/focus_field_parameters.field_parameters.wavelength #k is given in nm, the same as wavelength
        
        ex_lens=np.zeros((focus_field_parameters.custom_mask_parameters.divisions_phi,focus_field_parameters.custom_mask_parameters.divisions_theta),dtype=complex)
        ey_lens=np.zeros((focus_field_parameters.custom_mask_parameters.divisions_phi,focus_field_parameters.custom_mask_parameters.divisions_theta),dtype=complex)

        theta_values=np.linspace(0,focus_field_parameters.alpha,focus_field_parameters.custom_mask_parameters.divisions_theta)  #divisions of theta in which the trapezoidal 2D integration is done
        rho_values=np.sin(theta_values)*focus_field_parameters.f              #given by the sine's law
        # logger.debug(f"{min(rho_values)=}")
        # logger.debug(f"{max(rho_values)=}")
        phi_values_x=np.linspace(0,2*np.pi,focus_field_parameters.custom_mask_parameters.divisions_phi)   #divisions of phi in which the trapezoidal 2D integration is done
        phi_values_y=np.linspace(np.pi/2,5*np.pi/2,focus_field_parameters.custom_mask_parameters.divisions_phi)   #divisions of phi in which the trapezoidal 2D integration is done
        for i,phi in enumerate(phi_values_x):
            for j,rho in enumerate(rho_values):
                phase=maskfunction(rho,phi,focus_field_parameters.field_parameters.w0,focus_field_parameters.f,k)
                ex_lens[i,j]=phase
        for i,phi in enumerate(phi_values_y):
            for j,rho in enumerate(rho_values):
                phase=maskfunction(rho,phi,focus_field_parameters.field_parameters.w0,focus_field_parameters.f,k)
                ey_lens[i,j]=phase
        ex_lens*=np.cos(focus_field_parameters.field_parameters.polarization.gamma)*focus_field_parameters.field_parameters.I_0**0.5
        ey_lens*=np.sin(focus_field_parameters.field_parameters.polarization.gamma)*np.exp(1j*focus_field_parameters.field_parameters.polarization.beta)*focus_field_parameters.field_parameters.I_0**0.5
        
        logger.debug("Generating the field inciding on the lens...")
        # logger.debug("Input:")
        # parameters = (focus_field_parameters.alpha,focus_field_parameters.f,focus_field_parameters.custom_mask_parameters.divisions_phi,focus_field_parameters.custom_mask_parameters.divisions_theta,focus_field_parameters.field_parameters.polarization.gamma,focus_field_parameters.field_parameters.polarization.beta,focus_field_parameters.field_parameters.w0,focus_field_parameters.field_parameters.I_0,focus_field_parameters.field_parameters.wavelength)
        # parameter_names = ("alpha","f","divisions_phi","divisions_theta","gamma","beta","w0","I0","wavelength")
        # for parameter, name in zip(parameters, parameter_names):
        #     if isinstance(parameter, float) or isinstance(parameter, bool) or isinstance(parameter, int) or len(parameter) < 5:
        #         logger.debug(f"{name}: {parameter}")
        #     else:
        #         logger.debug(f"{name}: {np.mean(parameter)}")
        # logger.debug("")
        # logger.debug("Output:")
        # logger.debug(f"{np.mean(ex_lens)=}")
        # logger.debug(f"{np.mean(ey_lens)=}")
        # logger.debug("")
        
        return ex_lens,ey_lens

