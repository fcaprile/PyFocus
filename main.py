from front_end_ui import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtWidgets
from pyqtgraph.Qt import QtGui
from No_mask import *
from VPP import *
from plot import *
from SPP import *
from trapezoid2D_integration_functions import *
from maskfunction_goes_here import maskfunction

import numpy as np
import os
from matplotlib import pyplot as plt
import time
import sys
import configparser

class PP_simulator(QtGui.QMainWindow,Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        #Main functions:
        self.counter=0# counts the numer of simulations realized in this sesion
        self.setupUi(self)
        self.pushButton.clicked.connect(self.save_parameters)
        self.pushButton_2.clicked.connect(self.save_intensity)
        self.pushButton_3.clicked.connect(self.simulate)
        self.pushButton_4.clicked.connect(self.save_amplitudes)        
        self.pushButton_5.clicked.connect(self.selectsavefolder)   
        self.pushButton_6.clicked.connect(self.clear_plots)   
        self.radioButton.toggled.connect(self.lineEdit_19.setEnabled)
        self.radioButton.toggled.connect(self.lineEdit_22.setEnabled)
        
        #Functions dedicated to the saved files's names
        self.lineEdit_21.textEdited.connect(self.change_saving_name)
        self.comboBox.currentTextChanged.connect(self.change_default_name)
        self.modified_saving_name=False #Used later to avoid overwriting the changes in the name of the figure made by the user
        self.default_file_name='VPP mask simulation' #default name for simulations if nothing is changed
        self.previous_figure_name=''#used to aoid overwriting previous figures which have the same name
    # def selectmask(self):
    #     file_name = QFileDialog.getOpenFileName(self, 'Open file')[0]#,"Python files (*.py)")
    #     self.lineEdit_9.setText(file_name)
    #     from file_name import mask
    #     self.mask=mask
        
    def selectsavefolder(self):
        self.save_folder=QFileDialog.getExistingDirectory(self, "Select Directory")+'/'
        self.lineEdit_14.setText(self.save_folder)

    def get_parameters(self):
        NA=float(self.lineEdit.text()) #probablemente definir el self.NA era mejor, no se si conviene
        n=float(self.lineEdit_2.text())
        h=float(self.lineEdit_18.text())*10**6 #convert to nm
        w0=float(self.lineEdit_16.text())*10**6 #convert to nm
        f=h*n/NA #given by the sine condition
        wavelength=float(self.lineEdit_3.text())/n
        I0=float(self.lineEdit_15.text())
        laser_width=float(self.lineEdit_16.text())
        zp0=float(self.lineEdit_12.text())
        polarization=float(self.lineEdit_13.text())
        beta=float(self.lineEdit_11.text())
#        phip0=float(self.lineEdit_7.text())# por ahora no lo implemento, es el angulo 0 desde el que observas
        rsteps=float(self.lineEdit_5.text())
        zsteps=float(self.lineEdit_4.text())
        field_of_view=float(self.lineEdit_17.text())
        if self.radioButton.isChecked()==True:
            L=float(self.lineEdit_19.text())
            R=float(self.lineEdit_22.text())
        else:
            L='Not implemented'
            R='Not implemented'
        z_field_of_view=float(self.lineEdit_20.text())
        figure_name=str(self.lineEdit_21.text())
        if figure_name=='':
            figure_name='Phase mask simulation'
        self.parameters=(NA,n,h,f,w0,wavelength,polarization,beta,zp0,rsteps,zsteps,field_of_view,I0,L,R,z_field_of_view)
        return NA,n,h,f,w0,wavelength,polarization,beta,zp0,rsteps,zsteps,field_of_view,I0,L,R,z_field_of_view,figure_name
        
    def save_parameters(self):
        try:
            folder=str(self.lineEdit_14.text())        
            if str(folder)=='(leave like this for local folder)':
                folder=''
            name=self.figure_name
            if name.endswith('.txt')==False: #to ensure that if the user writes .txt on the name, it doesnt get repeated as .txt.txt
                name=name+' parameters'
            else:
                name=name[:-4]+' parameters'
                
            config = configparser.ConfigParser()
            config['Parameters'] = {
            'NA': self.parameters[0],
            'n': self.parameters[1],
            'focal distance (mm)': self.parameters[2]*10**-6,
            'incident beam radius (mm)': self.parameters[3]*10**-6,
            'wavelength at vacuum (nm)': self.parameters[5]*self.parameters[3],
            'arctan(ey ex) (degrees)': self.parameters[6],
            'phase diference (degrees)': self.parameters[7],
            'Axial distance from focus (nm)': self.parameters[8],
            'Radial resolution (nm)': self.parameters[9],
            'Axial resolution (nm)': self.parameters[10],
            'Radial field of view (nm)': self.parameters[11],
            'Axial field of view (nm)': self.parameters[15],
            'Laser intensity': self.parameters[12],
            'laser width (mm)': self.parameters[13],
            'Distance from phase plate to objective (mm)': self.parameters[14]}
            
            with open(folder+name+'.txt', 'w') as configfile:
                config.write(configfile)
            print('Parameters saved!')
                 
        except:
            print("Unexpected error:", sys.exc_info())
    def save_intensity(self):
        try:
            folder=str(self.lineEdit_14.text())      
            if str(folder)=='(leave like this for local folder)':
                folder=''
            name=self.figure_name
            if name.endswith('.txt')==False: #to ensure that if the user writes .txt on the name, it doesnt get repeated as .txt.txt
                name_XZ=name+' XZ plane intensity.txt'
                name_XY=name+' XY plane intensity.txt'
            else:
                name_XZ=name[:-4]+' XZ plane intensity.txt'
                name_XY=name[:-4]+' XY plane intensity.txt'
            np.savetxt(folder+name_XY,self.Ifield_xy, fmt='%.18g', delimiter='\t', newline=os.linesep)        
            np.savetxt(folder+name_XZ,self.Ifield_xz, fmt='%.18g', delimiter='\t', newline=os.linesep)        
            print('Intensity saved!')
        except:
            print("Unexpected error:", sys.exc_info())

    def save_amplitudes(self):
        try:
            folder=str(self.lineEdit_14.text())      
            if str(folder)=='(leave like this for local folder)':
                folder=''
            name=self.figure_name
            if name.endswith('.txt')==False: #to ensure that if the user writes .txt on the name, it doesnt get repeated as .txt.txt
                name_x=name+' X component amplitude.txt'
                name_y=name+' Y component amplitude.txt'
                name_z=name+' Z component amplitude.txt'
            else:
                name_x=name[:-4]+' X component amplitude.txt'
                name_y=name[:-4]+' Y component amplitude.txt'
                name_z=name[:-4]+' Z component amplitude.txt'
            np.savetxt(folder+name_x,self.amplitudes_xy[0], fmt='%.18g', delimiter='\t', newline=os.linesep)        
            np.savetxt(folder+name_y,self.amplitudes_xy[1], fmt='%.18g', delimiter='\t', newline=os.linesep)        
            np.savetxt(folder+name_z,self.amplitudes_xy[2], fmt='%.18g', delimiter='\t', newline=os.linesep)        
            print('Amplitudes saved!')
        except:
            print("Unexpected error:", sys.exc_info())
      
    def clear_plots(self):
        plt.close('all')
        
    def change_saving_name(self):
        self.modified_saving_name=True
        name=self.lineEdit_21.text()
        self.lineEdit_7.setText(str(name))
        self.lineEdit_8.setText(str(name))
        
    def change_default_name(self):
        selected=self.comboBox.currentIndex()
        if self.modified_saving_name==False:
            if selected==0:#VPP mask
                self.default_file_name='VPP mask simulation'
                self.lineEdit_21.setText(self.default_file_name)
            if selected==1:#No mask (gaussian beam)
                self.default_file_name='Gaussian beam simulation'
                self.lineEdit_21.setText(self.default_file_name)
            if selected==2:#Custom mask
                self.default_file_name='Custom mask simulation'
                self.lineEdit_21.setText(self.default_file_name)
            if selected==3:#Spiral Phase plate mask (SPP)
                self.default_file_name='Polarization mask simulation'
                self.lineEdit_21.setText(self.default_file_name)                
        
    def simulate(self):
        self.counter+=1
        NA,n,h,f,w0,wavelength,polarization,beta,zp0,rsteps,zsteps,field_of_view,I0,L,R,z_field_of_view,figure_name=self.get_parameters()
        
        #used to aoid overwriting previous figures which have the same name:
        if figure_name==self.default_file_name:
            figure_name+=' '+str(self.counter)
        if figure_name==self.previous_figure_name:
            figure_name+=' '+str(self.counter)
        self.previous_figure_name=figure_name        
        self.figure_name=figure_name #used as name for saving texts 
        alpha=np.arcsin(NA / n)
        selected=self.comboBox.currentIndex()
        try:                             
            if self.radioButton.isChecked()==False: #then no need to calculate the field at the entrance of the lens
                if selected==0: #VPP mask
                    #calculate field at the focal plane:
                    II1,II2,II3,II4,II5=VPP_integration(alpha,n,f,h,wavelength,zsteps,rsteps,field_of_view,z_field_of_view,w0)
                    fields_VPP=VPP_fields(II1,II2,II3,II4,II5,wavelength,I0,beta,polarization,zsteps,rsteps,field_of_view,z_field_of_view,0,n,f,zp0)                    
                    ex1,ey1,ez1,ex2,ey2,ez2=np.array(fields_VPP)

                    #plot the fields at the focus:
                    plot_XZ_XY(ex1,ey1,ez1,ex2,ey2,ez2,zsteps,rsteps,field_of_view,z_field_of_view,n,wavelength,I0,alpha,figure_name) #ex, ey and ez have tha values of the field on the XY plane
                    self.amplitudes_xy=(ex2,ey2,ez2)
                    self.Ifield_xz=np.abs(ex1)**2+np.abs(ey1)**2+np.abs(ez1)**2
                    self.Ifield_xy=np.abs(ex2)**2+np.abs(ey2)**2+np.abs(ez2)**2
        
                if selected==1: #No mask (gaussian beam)
                    #calculate field at the focal plane:
                    II1,II2,II3=no_mask_integration(alpha,n,f,wavelength,field_of_view,z_field_of_view,zsteps,rsteps,w0)
                    ex1,ey1,ez1,ex2,ey2,ez2=no_mask_fields(II1,II2,II3,wavelength,I0,beta,polarization,zsteps,rsteps,field_of_view,z_field_of_view,0,n,f,zp0)

                    #plot the fields at the focus:
                    plot_XZ_XY(ex1,ey1,ez1,ex2,ey2,ez2,zsteps,rsteps,field_of_view,z_field_of_view,n,wavelength,I0,alpha,figure_name) #ex, ey and ez have tha values of the field on the XY plane
                    self.amplitudes_xy=(ex2,ey2,ez2)
                    self.Ifield_xz=np.abs(ex1)**2+np.abs(ey1)**2+np.abs(ez1)**2
                    self.Ifield_xy=np.abs(ex2)**2+np.abs(ey2)**2+np.abs(ez2)**2        
                
                if selected==2: #Custom mask
                    # maskfunction=lambda theta, phi: np.exp(1j*phi) is the VPP mask function
                    #Usual parameters for integration precision:
                    #resolution for field at objective
                    resolution_theta=200
                    resolution_phi=200
                    #resolution for field near the focus
                    resolution_focus=int(field_of_view/rsteps)
                    resolution_z=int(z_field_of_view/zsteps)
                    
                    #pasage to mm:
                    wavelength*=10**-6
                    f*=10**-6
                    
                    print('Generating incident field')
                    time.sleep(0.5)
                    #calculate field at the entrance of the lens: (ex_lens,ey_lens) are 2 matrixes with the values of the incident amplitude for each value in phi,theta                   
                    ex_lens,ey_lens=generate_incident_field(maskfunction,alpha,f,resolution_phi,resolution_theta,R,beta,polarization,w0*10**-6,I0,wavelength)

                    #plot field at the entrance of the lens:
                    plot_in_cartesian(ex_lens,ey_lens,h*10**-6,alpha,f,figure_name)
                    
                    '''
                    #if the incident field comes from an outside matrix:
                    ex_lens=np.loadtxt(filename for X component of the field)
                    ey_lens=np.loadtxt(filename for Y component of the field)
                    '''
                    
                    #calculate field at the focal plane:
                    ex1,ey1,ez1,ex2,ey2,ez2=custom_mask_focus_field_XZ_XY(ex_lens,ey_lens,alpha,h,wavelength,z_field_of_view,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,field_of_view)

                    #plot the fields at the focus:
                    plot_XZ_XY(ex1,ey1,ez1,ex2,ey2,ez2,zsteps,rsteps,field_of_view,z_field_of_view,n,wavelength,I0,alpha,figure_name) #ex, ey and ez have tha values of the field on the XY plane
                    self.amplitudes_xy=(ex2,ey2,ez2)
                    self.Ifield_xz=np.abs(ex1)**2+np.abs(ey1)**2+np.abs(ez1)**2
                    self.Ifield_xy=np.abs(ex2)**2+np.abs(ey2)**2+np.abs(ez2)**2
                '''
                if selected==3: #Spiral Phase plate mask (SPP), removed
                    #calculate field at the focal plane:
                    ex1,ey1,ez1,ex2,ey2,ez2=SPP_simulation(polarization,0,beta,zp0,I0,alpha,n,f,h,wavelength,zsteps,rsteps,field_of_view,z_field_of_view,w0)

                    #plot the fields at the focus:
                    plot_XZ_XY(ex1,ey1,ez1,ex2,ey2,ez2,zsteps,rsteps,field_of_view,z_field_of_view,n,wavelength,I0,alpha,figure_name) #ex, ey and ez have tha values of the field on the XY plane
                    self.amplitudes_xy=(ex2,ey2,ez2)
                    self.Ifield_xz=np.abs(ex1)**2+np.abs(ey1)**2+np.abs(ez1)**2
                    self.Ifield_xy=np.abs(ex2)**2+np.abs(ey2)**2+np.abs(ez2)**2
                '''
            else:
                if selected==0: #VPP mask
                    #calculate field at the entrance of the lens:                    
                    E_rho,Ex,Ey=VPP_pre_focus_opt(beta,polarization,1500,R,L,I0,wavelength*10**-6,2.2*R,w0*10**-6,2000,20,figure_name=figure_name)

                    #calculate field at the focal plane:
                    II1,II2,II3,II4,II5=VPP_integration_with_L(alpha,n,f,R,wavelength,zp0,zsteps,rsteps,field_of_view,w0,E_rho,20)
                    fields_VPP=VPP_fields_with_L(II1,II2,II3,II4,II5,wavelength,I0,beta,polarization,zsteps,rsteps,field_of_view,0,n,f,zp0)
                    ex2,ey2,ez2=np.array(fields_VPP)

                    #plot the fields at the focus:
                    plot_XY(ex2,ey2,ez2,rsteps,field_of_view,n,wavelength,I0,alpha,f,figure_name)
                    self.amplitudes_xy=(ex2,ey2,ez2)
                    self.Ifield_xy=np.abs(ex2)**2+np.abs(ey2)**2+np.abs(ez2)**2
                if selected==1:
                    print('Propagation for field at objective only for VPP and used defined masks')
                if selected==2: #Custom mask
                    # maskfunction=lambda theta, phi: np.exp(1j*phi) is the VPP mask function
                    #Usual parameters for integration precision:
                    #Number of divisions for trapezoidal integration for the field at the objective
                    N_rho=200 #if distances between VPP and objective larger than 300mm are used, a smaller number of divisions still shields a proper integration
                    N_phi=200
                    #resolution for field at the objective
                    resolution_theta=100
                    resolution_phi=100
                    #resolution for field near the focus
                    resolution_focus=int(field_of_view/rsteps)
                    resolution_z=int(z_field_of_view/zsteps)
                    
                    #pasage to mm:
                    wavelength*=10**-6
                    f*=10**-6
                    
                    #make field at the entrance of the lens:
                    ex_lens,ey_lens,I_cartesian,Ex_cartesian,Ey_cartesian=custom_mask_objective_field(beta,polarization,resolution_theta,resolution_phi,N_rho,N_phi,alpha,maskfunction,R,L,I0,wavelength,w0*10**-6,figure_name,plot=True)
                    
                    #calculate field at the focal plane:
                    ex1,ey1,ez1,ex2,ey2,ez2=custom_mask_focus_field_XZ_XY(ex_lens,ey_lens,alpha,h,wavelength,z_field_of_view,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,field_of_view)

                    #plot the fields:
                    plot_XZ_XY(ex1,ey1,ez1,ex2,ey2,ez2,zsteps,rsteps,field_of_view,z_field_of_view,n,wavelength,I0,alpha,figure_name) #ex, ey and ez have tha values of the field on the XY plane
                    self.amplitudes_xy=(ex2,ey2,ez2)
                    self.Ifield_xz=np.abs(ex1)**2+np.abs(ey1)**2+np.abs(ez1)**2
                    self.Ifield_xy=np.abs(ex2)**2+np.abs(ey2)**2+np.abs(ez2)**2
        except:
            print("Unexpected error:", sys.exc_info())
    def closeEvent(self, event, *args, **kwargs):

        print('Module closed')        
        super().closeEvent(*args, **kwargs)

        
        
if __name__ == '__main__':
    
    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()
        
    print('Module running')    
    gui = PP_simulator()
    gui.show()