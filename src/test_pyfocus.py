# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 09:45:35 2021

@author: Fernando Caprile
"""

# import sys, os
# sys.path.append('../')
# PACKAGE_PARENT = '..'
# SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
# sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from PyFocus import sim, plot
import numpy as np
from matplotlib import pyplot as plt


NA=0.65
n=1.5
h=3
w0=50
wavelength=532
I0=1

gamma=45
beta=90

z=0
x_steps=300
z_steps=600
x_steps=30
z_steps=200
x_range=900
z_range=2000
figure_name='Example'

L=''
R=''
ds=''
z_int=''

n=1.5
# n=np.array((1.5,1.5))
ds= np.array((np.inf,np.inf))
z_int=0

parameters=np.array((NA, n, h, w0, wavelength, gamma, beta, z, x_steps, z_steps, x_range, z_range, I0, L, R, ds, z_int, figure_name), dtype=object)
divisions_theta=100
divisions_phi=100

entrance_field=lambda rho,phi,w0,f,k:1
custom_mask=lambda rho,phi,w0,f,k:1#np.exp(1j*phi)
fields=sim.custom(entrance_field,custom_mask,False,False,*parameters,divisions_theta,divisions_phi,plot_Ei=True)
# fields = sim.no_mask(False,False,*parameters)
# fields = sim.VP(False,False,*parameters)
fig1,fig2=plot.plot_XZ_XY(*fields,x_range,z_range,figure_name)
