import numpy as np
import matplotlib.pyplot as plt

mapping_factor = 2
input_file_name = 'input.txt'
output_file_name = 'output.txt'

def coordinate_change(field, h, alpha, f, mapping_resolution_factor):
    x_steps = np.shape(field)[0]
    x_values = np.linspace(-h,h,x_steps)
    y_values = np.linspace(-h,h,x_steps)
    mapping_res_phi = x_steps* mapping_resolution_factor
    mapping_res_theta = x_steps* mapping_resolution_factor
    '''Convierte el campo incidente de coordenadas cartesianas a las coordenadas polares usadas en la integral'''
    mapped_field=np.zeros((mapping_res_phi,mapping_res_theta),dtype=complex)

    theta_values=np.linspace(0,alpha,mapping_res_theta)  #divisions of theta in which the trapezoidal 2D integration is done
    rho_values=np.sin(theta_values)*f              #given by the sine's law
    phi_values=np.linspace(0,2*np.pi,mapping_res_phi)   #divisions of phi in which the trapezoidal 2D integration is done
    for i,phi in enumerate(phi_values):
        for j,rho in enumerate(rho_values):
            #Coordenadas reales dado rho, phi
            real_x = rho*np.cos(phi)
            real_y = rho*np.sin(phi)
            #Valores del campo lo más cercano a las coordenadas dadas
            mapped_x_index = np.argmin(np.abs(np.array(x_values)-real_x))
            mapped_y_index = np.argmin(np.abs(np.array(y_values)-real_y))
            mapped_field[i,j]=field[mapped_x_index, mapped_y_index]
    return mapped_field

NA=1.4
n=1.5
alpha=np.arcsin(NA / n)
h=3
f=h*n/NA  

def from_amplitude_to_phase(amplitude_matrix):
    a,b = np.shape(amplitude_matrix)
    aux = np.zeros((a,b),dtype=complex)
    for i in range(a):
        for j in range(b):
            aux[i,j]=np.exp(1j*amplitude_matrix[i,j])
    
    return aux


input_matrix = np.loadtxt(input_file_name, dtype=complex)
shifted_matrix = coordinate_change(from_amplitude_to_phase(input_matrix), h, alpha, f,mapping_factor)
np.savetxt(output_file_name,shifted_matrix)

fig, ax = plt.subplots(1,2)
ax[0].imshow(np.angle(input_matrix))
ax[1].imshow(np.angle(shifted_matrix))

