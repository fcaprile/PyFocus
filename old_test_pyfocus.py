# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 09:45:35 2021

@author: Fernando Caprile
"""

from PyFocus import sim
from PyFocus.src.plot import plot
import numpy as np
from matplotlib import pyplot as plt


NA=1.4
n=1.5
h=3
w0=5
wavelength=640
I0=1

gamma=0
beta=0

z=0
x_steps=2
z_steps=5
x_range=800
z_range=1500
figure_name='Example'

L=''
R=''
ds=''
z_int=''

parameters=np.array((NA, n, h, w0, wavelength, gamma, beta, z, x_steps, z_steps, x_range, z_range, I0, L, R, ds, z_int, figure_name), dtype=object)


divisions_theta=30
divisions_phi=30

entrance_field=lambda rho,phi,w0,f,k:1
custom_mask=lambda rho,phi,w0,f,k:1
fields=sim.custom(entrance_field,custom_mask,False,False,*parameters,divisions_theta,divisions_phi)
fields2=np.copy(fields)
# Ez=fields2[-1]

plt.close('all')
fig1,fig2=plot.plot_XZ_XY(*fields2,x_range,z_range,figure_name)

