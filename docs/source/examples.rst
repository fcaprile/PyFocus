Examples using PyFocus User Interface
===================================================

First we import the user interface class and define an instance:

.. code-block:: python

   from PyFocus import user_interface 
   
   ui=user_interface.UI()

To show the interface we use the function "show":

.. code-block:: python

   ui.show()

If this example does not work, please contact us. You can continue with the rest of the examples using PyFocus's high-level functions, or use the previously mentioned "PyFocus.exe" windows executable. 

Examples using PyFocus in custom codes
===================================================

First, we define the parameters of the simulation:

        :NA (float): Numerical aperture        
        :n (float or array): Type depends on interface: float for no interface, array for interface. Refraction index for the medium of the optical system.       
        :h (float): Radius of aperture of the objective lens(mm)
                
        :w0 (float): Radius of the incident gaussian beam (mm)
        
        :wavelength (float): Wavelength in vacuum
        
        :gamma (float): Parameter that determines the polarization, arctan(ey/ex) (gamma=45, beta=90 gives right circular polarization and a donut shape)
        
        :beta (float): parameter that determines the polarization, phase difference between ex and ey (gamma=45, beta=90 gives right circular polarization and a donut shape)
        
        :z (float): Axial position for the XY plane (nm)
        
        :x_steps (float): Resolution in the X or Y coordinate for the focused field (nm)
        
        :z_steps (float): Resolution in the axial coordinate (Z) for the focused field (nm)
        
        :x_range (float): Field of view in the X or Y coordinate in which the focused field is calculated (nm)
        
        :z_range (float): field of view in the axial coordinate (z) in which the focused field is calculated (nm)
        
        :I_0 (float): Incident field intensity (mW/cm^2)
        
        :L (float, optional): Distance between phase mask and objective lens (mm), only used if propagation=True
        
        :R (float, optional): Phase mask radius (mm), only used if propagation=True
        
        :ds (array, optional): Thickness of each interface in the multilayer system (nm), must be given as a numpy array with the first and last values a np.inf. Only used if interface=True
        
        :z_int (float, optional): Axial position of the interphase. Only used if interface=True
        
        :figure_name (string, optional): Name for the images of the field. Also used as saving name if using the UI    

.. code-block:: python

   NA=1.4 
   n=1.5
   h=3
   w0=5
   wavelength=640
   I0=1

We define an incident beam of left circular polarization:

.. code-block:: python

   gamma=45
   beta=90

Next, we define the range in which we realize the simulation, the resolution and the name of the resulting figures:

.. code-block:: python

   z=0
   x_steps=5
   z_steps=10
   x_range=1000
   z_range=2000
   figure_name='Example'

For the first examples we will neglect the propagation of the incident field and won't simulate an interface and so the parameters L, R, ds and z_int won't be used by the code. For completion, we will define them as '':

.. code-block:: python

   L=''
   R=''
   ds=''
   z_int=''

To simplify the code, we define the "parameters" array:

.. code-block:: python

   parameters=np.array((NA, n, h, w0, wavelength, gamma, beta, z, x_steps, z_steps, x_range, z_range, I0, L, R, ds, z_int, figure_name), dtype=object)


Gaussian beam without modulation
--------------------------------

Simulation of a focused gaussian beam without phase modulation by using the "no_mask" function. We obtain the "fields" tuple, which contains 6 arrays with the resulting field, and then we plot them using the "plot_XZ_XY" function. This function returns a figure of the total intensity and polarization, here called fig1 and a figure of the intensity of each cartesian component:

.. code-block:: python

   fields=sim.no_mask(False,False,*parameters) 
   fig1,fig2=plot.plot_XZ_XY(*fields,x_range,z_range,figure_name)

VP mask modulation
==================

Simulation of a focused gaussian beam with VP modulation by using the "VP" function. Remember that the 2 first ariables are boolean parameters that define if we simulate the propagation of the incident field and if there is an interface.

.. code-block:: python

   fields=sim.VP(False,False,*parameters) 
   fig1,fig2=plot.plot_XZ_XY(*fields,x_range,z_range,figure_name)


Propagation and modulation by a VP mask
=======================================

To simulate the propagation of the incident field from the phase mask to the objective lens, we redefine the parameters L and R and set the propagation variable of VP to True:

.. code-block:: python

   L=1000
   R=5
   parameters=np.array((NA, n, h, w0, wavelength, gamma, beta, z, x_steps, z_steps, x_range, z_range, I0, L, R, ds, z_int, figure_name), dtype=object)

   fields=sim.VP(True,False,*parameters)
   fig1,fig2=plot.plot_XZ_XY(*fields,x_range,z_range,figure_name)

Interface and modulation by a VP mask
=======================================

To simulate a glass-water interface located at the focal plane, we redefine the parameters n, ds and z_int and set the interface variable of VP to True:

.. code-block:: python

   n=np.array((1.5,1.33))
   ds= np.array((np.inf,np.inf))
   z_int=0
   parameters=np.array((NA, n, h, w0, wavelength, gamma, beta, z, x_steps, z_steps, x_range, z_range, I0, L, R, ds, z_int, figure_name), dtype=object)

   fields=sim.VP(False,True,*parameters)
   fig1,fig2=plot.plot_XZ_XY(*fields,x_range,z_range,figure_name)

Multilayer system
-----------------

For a multilayer system, we add 2 more layers of refraction index 0.14+3.55j and 1.54, and thicknesses 44 and 24:

.. code-block:: python

   n=np.array((1.5,0.14+3.55j,1.54,1.33))
   ds= np.array((np.inf,44,24,np.inf))
   z_int=0
   parameters=np.array((NA, n, h, w0, wavelength, gamma, beta, z, x_steps, z_steps, x_range, z_range, I0, L, R, ds, z_int, figure_name), dtype=object)

   fields=sim.VP(False,True,*parameters)
   fig1,fig2=plot.plot_XZ_XY(*fields,x_range,z_range,figure_name)


Custom mask examples
=====================

Missalignment
--------------

First, we simulate the displacement of the gaussian beam and the VP mask. We define the auxiliar variables rho2 and phi2, which correspond to the displaced coordinates. We set the displacement to 0.4 times the aperture radius of the objective lens. 

.. code-block:: python

   d=0.4*h
   rho2=lambda rho,phi:(rho**2+d**2-2*rho*d*np.cos(phi))**0.5
   phi2=lambda rho,phi:np.arctan2(rho*np.sin(phi),rho*np.cos(phi)-d)

Then we define the mask function and begin the simulation. We set the 2D integration divisions to be 200 for both variables:

.. code-block:: python

   entrance_field=lambda rho,phi,w0,f,k: np.exp(-(rho2(rho,phi)/w0)**2)
   custom_mask=lambda rho,phi,w0,f,k: np.exp(1j*phi2(rho,phi))
   divisions_theta=200
   divisions_phi=200
   fields=sim.custom(entrance_field,custom_mask,False,False,*parameters,divisions_theta,divisions_phi)
   fig1,fig2=plot.plot_XZ_XY(*fields,x_range,z_range,figure_name)

Inclination (tilt)
-------------------

For inclination in an angle of 3.11*10**-5 radians:

.. code-block:: python

   angle=3.11*10**-5
   
   entrance_field=lambda rho,phi,w0,f,k: np.exp(-(rho/w0)**2)
   custom_mask=lambda rho,phi,w0,f,k: np.exp(1j*(phi+k*np.sin(ang)*rho*np.cos(phi)))

   divisions_theta=200
   divisions_phi=200
   fields=sim.custom(entrance_field,custom_mask,False,False,*parameters,divisions_theta,divisions_phi)
   fig1,fig2=plot.plot_XZ_XY(*fields,x_range,z_range,figure_name)

TIRF
-----

After setting a water-glass interface, to simulate TIRF with modulation of a VP mask we define the mask function the following way:

.. code-block:: python

   n=np.array((1.5,1.33))
   ds= np.array((np.inf,np.inf))
   z_int=0
   parameters=np.array((NA, n, h, w0, wavelength, gamma, beta, z, x_steps, z_steps, x_range, z_range, I0, L, R, ds, z_int, figure_name), dtype=object)

   theta_crit=np.arcsin(n[1]/n[0])
   entrance_field=lambda rho,phi,w0,f,k: np.exp(-(rho/w0)**2)
   def custom_mask(rho,phi,w0,f,k):
       if rho>f*np.sin(theta_crit):
           return np.exp(1j*phi)
       else:
           return 0

   fields=sim.custom(entrance_field,custom_mask,False,True,*parameters,200,200)
   fig1,fig2=plot.plot_XZ_XY(*fields,x_range,z_range,figure_name)















