#user interphase packages
from front_end_ui import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog
from qtpy import QtWidgets
from pyqtgraph.Qt import QtGui
import qdarkstyle

#custom made integration functions
from No_mask import no_mask_integration, no_mask_fields
from VPP import VPP_integration, VPP_fields, VPP_fraunhofer, VPP_integration_with_propagation, VPP_fields_with_propagation
from plot import plot_XZ_XY, plot_XY
from trapezoid2D_integration_functions import generate_incident_field, plot_in_cartesian, custom_mask_objective_field, custom_mask_focus_field_XZ_XY
from trapezoid2D_with_interphase_integration_functions import interphase_custom_mask_focus_field_XZ_XY
from maskfunction_goes_here import maskfunction

#usual packages
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import sys
import configparser


class PyFocus(QtGui.QMainWindow,Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        print('PyFocus is running')    
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        app = QtGui.QApplication.instance()
        app.setStyleSheet(qdarkstyle.load_stylesheet())#set a dark style

        #Main functions:
        self.counter=0# counts the numer of simulations realized in this sesion, to avoid overwriting save files with the same name
        self.setupUi(self)
        self.pushButton.clicked.connect(self.save_parameters)
        self.pushButton_2.clicked.connect(self.save_intensity)
        self.pushButton_3.clicked.connect(self.simulate)
        self.pushButton_4.clicked.connect(self.save_amplitudes)        
        self.pushButton_5.clicked.connect(self.selectsavefolder)   
        self.pushButton_6.clicked.connect(self.clear_plots)   
        self.radioButton.toggled.connect(self.lineEdit_19.setEnabled)#enable modifying parameters for simulation of the field inciding on the lens
        self.radioButton.toggled.connect(self.lineEdit_22.setEnabled)
        self.radioButton_2.toggled.connect(self.lineEdit_23.setEnabled)#enable modifying parameters for an interphase
        self.radioButton_2.toggled.connect(self.lineEdit_24.setEnabled)
        self.radioButton_2.toggled.connect(self.lineEdit_25.setEnabled)
        self.radioButton_2.toggled.connect(self.lineEdit_2.clear)
        self.radioButton_2.toggled.connect(self.lineEdit_2.setDisabled)
        
        #Functions dedicated to the saved files's names
        self.lineEdit_21.textEdited.connect(self.change_saving_name)
        self.comboBox.currentTextChanged.connect(self.change_default_name)
        self.modified_saving_name=False #Used later to avoid overwriting the changes in the name of the figure made by the user
        self.default_file_name='VPP mask simulation' #default name for simulations if nothing is changed
        self.previous_figure_name=''#used to aoid overwriting previous figures which have the same name
        
    def selectsavefolder(self):
        '''
        Allows selecting a save folder for the txt files
        '''
        self.save_folder=QFileDialog.getExistingDirectory(self, "Select Directory")+'/'
        self.lineEdit_14.setText(self.save_folder)

    def get_parameters(self):
        '''
        Read parameters from the UI
        '''
        NA=float(self.lineEdit.text()) #numerical aperture
        n=float(self.lineEdit_2.text())#medium's density
        h=float(self.lineEdit_18.text())*10**6 #radius of aperture converted to nm
        w0=float(self.lineEdit_16.text())*10**6 #incident gaussian beam's radius, converted to nm
        f=h*n/NA #given by the sine's law
        wavelength=float(self.lineEdit_3.text())#wavelength if there is not an interphase between 2 mediums
        I0=float(self.lineEdit_15.text())#maximum intensity for the incident beam (|A|^2)
        zp0=float(self.lineEdit_12.text())#axial distance in wich to calculate the XY plane (z=cte=zp0 is the plane)
        gamma=float(self.lineEdit_11.text()) #arctan(ey/ex)
        beta=float(self.lineEdit_13.text()) #Delta phase betheen ey and ey, gamma=45 and beta=90 give right circular poalarizacion and a donut, gamma=45 and beta=-90 give left polarization and a maximum of intensity in the focused field
        rsteps=float(self.lineEdit_5.text())#radial pixel size
        zsteps=float(self.lineEdit_4.text())#axial pixel size
        field_of_view=float(self.lineEdit_17.text())#radial distance for the simulation
        if self.radioButton.isChecked()==True:#L and R are parameters for calculating the inciding beam with fraunhoffer's formula, if the propagation is depreciated then this parameters ar saved as ''not implemented''
            L=float(self.lineEdit_19.text())
            R=float(self.lineEdit_22.text())
        else:
            L='Not implemented'
            R='Not implemented'
        z_field_of_view=float(self.lineEdit_20.text())#axial distance for the simulation
        figure_name=str(self.lineEdit_21.text())#name for the ploted figures and the save files
        if figure_name=='': #to always have a figure name
            figure_name='Phase mask simulation'
        if self.radioButton_2.isChecked()==True:#if the interphase simulation is not checked, this parameters ar saved as ''not implemented''
            try:
                self.zint=float(self.lineEdit_25.text())#axial distance of the interphase (negatie means before the focal plane)
                self.ns=np.array([float(n) for n in self.lineEdit_23.text().split(',')])#array containing the refraction indexes of all mediums
                self.ds=[]#array cointaining the thickness of all mediums
                self.ds.append(np.inf)
                n='Interphases'
                if not self.lineEdit_24.text()=='':
                    for d in [complex(s) for s in self.lineEdit_24.text().split(',')]:
                        self.ds.append(d)
                self.ds.append(np.inf)
                self.ds=np.array(self.ds)
            except:
                print('wrong format, explample of correct format: ns=1.5,1.2,1.4,1j*1.2,1.33, ds=5,10,2)',sys.exc_info())
        else:
            self.zint='Not implemented'
            self.ns='Not implemented'
            self.ds='Not implemented'
        self.parameters=NA,n,h,f,w0,wavelength,gamma,beta,zp0,rsteps,zsteps,field_of_view,z_field_of_view,I0,L,R,figure_name#used as an extra variable to save the parameters
        return NA,n,h,f,w0,wavelength,gamma,beta,zp0,rsteps,zsteps,field_of_view,z_field_of_view,I0,L,R,figure_name
        
    def save_parameters(self):
        '''
        Save a txt file with the parameters used for the simulation with the UI, parameters can be strings
        '''
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
            'refraction indexes': self.parameters[1],
            'Aperture radius (mm)':self.parameters[2]*10**-6,
            'focal distance (mm)': self.parameters[3]*10**-6,
            'incident beam radius (mm)': self.parameters[4]*10**-6,
            'wavelength at vacuum (nm)': self.parameters[5],
            'gamma, arctan(ey ex) (degrees)': self.parameters[6],
            'beta, Delta phase (degrees)': self.parameters[7],
            'Axial distance from focus for the XY plane (nm)': self.parameters[8],
            'Radial pixel size (nm)': self.parameters[9],
            'Axial pixel size (nm)': self.parameters[10],
            'Radial field of view (nm)': self.parameters[11],
            'Axial field of view (nm)': self.parameters[12],
            'Laser intensity (kW/cm^2)': self.parameters[13],
            'Distance from phase plate to objective (mm)': self.parameters[14],
            'Phase mask radius (mm)': self.parameters[15],
            'Interphase axial position': self.zint,
            'Interphase indexes': self.ns,
            'Interphase layer thickness': self.ds}
            with open(folder+name+'.txt', 'w') as configfile:
                config.write(configfile)
            print('Parameters saved!')
                 
        except:
            print("Unexpected error:", sys.exc_info())
            
    def save_intensity(self):
        '''
        Save a txt file with the intensity on the simulated planes
        '''
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
        '''
        Save a txt file with the amplitude of each cartesian component on the simulated planes
        '''
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
        '''
        Clear all existing figures
        '''
        plt.close('all')
        
    def change_saving_name(self):
        '''
        Change saving name for the UI, also modifying the 
        '''
        self.modified_saving_name=True
        self.name=self.lineEdit_21.text()
        
    def change_default_name(self):
        '''
        Quality of life modification, if a new mask is selected and the name has never been modified, then change the default simulation name 
        '''
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
                
    def VPP(self,propagation,NA,n,h,f,w0,wavelength,gamma,beta,zp0,rsteps,zsteps,field_of_view,z_field_of_view,I0,L,R,figure_name,ns='Not implemented',ds='Not implemented'):
        '''
        Simulate the field obtained by focusing a gaussian beam modulated by a VPP mask 
        propagation=True calculates and plots the field inciding on the lens by fraunhofer's difractin formula
        propagation=False calculates the field inciding on the lens depreciating the propagation
        ns and ds are parameters given in case of an interphase, otherwise they must be left as empty        
        
        Returns the ampitude of each component on the y=0 plane (XZ) and z=cte (XY) with the constant given by the user on ''axial distance from focus'', named on the code as zp0
        Returns the amplitude as ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY with for example ex the amplitude matrix (each place of the matrix is a different spacial position) of the X component and XZ or XY the plane in which they where calculated
        '''
        
        alpha=np.arcsin(NA / n)
        
        if ns=='Not implemented' and ds=='Not implemented':#no interphase
            if propagation==False:
                II1,II2,II3,II4,II5=VPP_integration(alpha,n,f,w0,wavelength/n,rsteps,zsteps,field_of_view,z_field_of_view)
                ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=VPP_fields(II1,II2,II3,II4,II5,wavelength,I0,gamma,beta,rsteps,zsteps,field_of_view,z_field_of_view,0,n,f,zp0)                        
            else:
                E_rho,Ex,Ey=VPP_fraunhofer(gamma,beta,1000,R,L,I0,wavelength*10**-6/n,2.2*R,w0*10**-6,2000,20,figure_name=figure_name)
                '''
                Funny enough, doing a 2D integration with the trapesoid method will be faster since the function converges reeeeally slowly, sorry if this is confusing
                '''
                #Usual parameters for integration precision:
                #resolution for field at objective
                resolution_theta=200
                resolution_phi=200
                #pasage to mm:
                wavelength*=10**-6
                f*=10**-6

                ex_lens=np.zeros((resolution_phi,resolution_theta),dtype=complex)
                ey_lens=np.zeros((resolution_phi,resolution_theta),dtype=complex)
            
                theta_values=np.linspace(0,alpha,resolution_theta)  #divisions of theta in which the trapezoidal 2D integration is done
                rhop_values=np.sin(theta_values)*f              #given by the sine's law
                phip_values=np.linspace(0,2*np.pi,resolution_phi)   #divisions of phi in which the trapezoidal 2D integration is done
                for i,phip in enumerate(phip_values):
                    for j,rhop in enumerate(rhop_values):
                        ex_lens[i,j]=E_rho(rhop)*np.exp(1j*phip)
                        ey_lens[i,j]=E_rho(rhop)*np.exp(1j*phip)
                ex_lens*=np.cos(gamma*np.pi/180)*I0**0.5
                ey_lens*=np.sin(gamma*np.pi/180)*np.exp(1j*beta*np.pi/180)*I0**0.5

                
                #resolution for field near the focus
                resolution_focus=int(field_of_view/rsteps)
                resolution_z=int(z_field_of_view/zsteps)
                
                #smaller numbers give faster integration times
                
                #calculate field at the focal plane:
                ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=custom_mask_focus_field_XZ_XY(ex_lens,ey_lens,alpha,h,wavelength/n,z_field_of_view,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,field_of_view)
        else:#interphase is given
            maskfunction=lambda theta, phi,w0,f,wavelength: np.exp(1j*phi)# is the VPP mask function
            #Usual parameters for integration precision:
            #resolution for field at objective
            resolution_theta=200
            resolution_phi=200
            
            #resolution for field near the focus
            resolution_focus=int(field_of_view/rsteps)
            resolution_z=int(z_field_of_view/zsteps)
            
            #smaller numbers give faster integration times
            
            #pasage to mm:
            wavelength*=10**-6
            f*=10**-6
            
            if propagation==False:
                #calculate field inciding on the lens by multiplying the phase mask's function and the gaussian amplitude
                #(ex_lens,ey_lens) are 2 matrixes with the values of the incident amplitude for each value in phi,theta                   
                ex_lens,ey_lens=generate_incident_field(maskfunction,alpha,f,resolution_phi,resolution_theta,h,gamma,beta,w0*10**-6,I0,wavelength/n)
                #plot field at the entrance of the lens:
                #since ex_lens and ey_lens are given in theta and phi coordinates, the have to be transformed to cartesian in order to be ploted, hence the name of this function
            else:
                #calculate field inciding on the lens by fraunhofer's difraction formula
                #N_rho and N_phi are the number of divisions for fraunhoffer's integral by the trapecium's method
                N_rho=400
                N_phi=400                
                ex_lens,ey_lens,I_cartesian,Ex_cartesian,Ey_cartesian=custom_mask_objective_field(gamma,beta,resolution_theta,resolution_phi,N_rho,N_phi,alpha,maskfunction,R,L,I0,wavelength/n,w0*10**-6,figure_name,plot=True)
    
            #calculate field at the focal plane:
            ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=interphase_custom_mask_focus_field_XZ_XY(self.ns,self.ds,ex_lens,ey_lens,alpha,h,wavelength,self.zint,z_field_of_view,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,field_of_view,x0=0,y0=0,z0=0,plot_plane='X')
        
        self.amplitudes_xy=(ex_XY,ey_XY,ez_XY)
        self.Ifield_xy=np.abs(ex_XY)**2+np.abs(ey_XY)**2+np.abs(ez_XY)**2#used to generate the save file
        self.Ifield_xz=np.zeros((2,2))#since it is not calculated i return a matrix of zeros so there is no error while trying to save the data
        return ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY
 
    def No_mask(self,NA,n,h,f,w0,wavelength,gamma,beta,zp0,rsteps,zsteps,field_of_view,z_field_of_view,I0,L,R,figure_name,ns='Not implemented',ds='Not implemented'):
        '''
        Simulate the field obtained by focusing a gaussian beam without being modulated in phase
        Since there is no phase mask, propagation is not a parameter
        
        Returns the ampitude of each component on the y=0 plane (XZ) and z=cte (XY) with the constant given by the user on ''axial distance from focus'', named on the code as zp0
        Returns the amplitude as ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY with for example ex the amplitude matrix (each place of the matrix is a different spacial position) of the X component and XZ or XY the plane in which they where calculated
        '''
        alpha=np.arcsin(NA / n)

        if ns=='Not implemented' and ds=='Not implemented':#no interphase
            II1,II2,II3=no_mask_integration(alpha,n,f,w0,wavelength/n,field_of_view,z_field_of_view,zsteps,rsteps)
            ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=no_mask_fields(II1,II2,II3,wavelength/n,I0,beta,gamma,zsteps,rsteps,field_of_view,z_field_of_view,0,n,f,zp0)
    
        else:#interphase
            maskfunction=lambda theta, phi,w0,f,wavelength: 1# is the nule mask function
            resolution_theta=200
            resolution_phi=200
            resolution_focus=int(field_of_view/rsteps)
            resolution_z=int(z_field_of_view/zsteps)
            wavelength*=10**-6
            f*=10**-6            
            ex_lens,ey_lens=generate_incident_field(maskfunction,alpha,f,resolution_phi,resolution_theta,h,gamma,beta,w0*10**-6,I0,wavelength)
            ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=interphase_custom_mask_focus_field_XZ_XY(self.ns,self.ds,ex_lens,ey_lens,alpha,h,wavelength,self.zint,z_field_of_view,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,field_of_view,x0=0,y0=0,z0=0,plot_plane='X')

        self.amplitudes_xy=(ex_XY,ey_XY,ez_XY)#used to generate the save file
        self.Ifield_xz=np.abs(ex_XZ)**2+np.abs(ey_XZ)**2+np.abs(ez_XZ)**2#used to generate the save file
        self.Ifield_xy=np.abs(ex_XY)**2+np.abs(ey_XY)**2+np.abs(ez_XY)**2#used to generate the save file

        return ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY

    def Custom(self,propagation,NA,n,h,f,w0,wavelength,gamma,beta,zp0,rsteps,zsteps,field_of_view,z_field_of_view,I0,L,R,figure_name,ns='Not implemented',ds='Not implemented'):
        '''
        Simulate the field obtained by focusing a gaussian beam modulated by a custom phase mask
        The amplitude term of a gaussian beam is already multiplyed to the integral despite the phase mask used, if this is not desired w0=100 (a big number) makes this term essentially 1
        
        propagation=True calculates and plots the field inciding on the lens by fraunhofer's difractin formula
        propagation=False calculates the field inciding on the lens depreciating the propagation
       
        Returns the ampitude of each component on the y=0 plane (XZ) and z=cte (XY) with the constant given by the user on ''axial distance from focus'', named on the code as zp0
        Returns the amplitude as ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY with for example ex the amplitude matrix (each place of the matrix is a different spacial position) of the X component and XZ or XY the plane in which they where calculated

        If the incident field comes from an outside matrix obtaned in some other way:
        ex_lens=np.loadtxt(filename for X component of the field)
        ey_lens=np.loadtxt(filename for Y component of the field)
        remember that each value of the matrix is given by a coordinate for theta and phi
        '''
        
        alpha=np.arcsin(NA / n)
        # maskfunction=lambda theta, phi: np.exp(1j*phi) is the VPP mask function
        #Usual parameters for integration precision:
        #resolution for field at objective
        resolution_theta=200
        resolution_phi=200
        
        #resolution for field near the focus
        resolution_focus=int(field_of_view/rsteps)
        resolution_z=int(z_field_of_view/zsteps)
        
        #smaller numbers give faster integration times
        
        #pasage to mm:
        wavelength*=10**-6
        f*=10**-6
                
        if propagation==False:
            #calculate field inciding on the lens by multiplying the phase mask's function and the gaussian amplitude
            #(ex_lens,ey_lens) are 2 matrixes with the values of the incident amplitude for each value in phi,theta                   
            ex_lens,ey_lens=generate_incident_field(maskfunction,alpha,f,resolution_phi,resolution_theta,h,gamma,beta,w0*10**-6,I0,wavelength/n)
            #plot field at the entrance of the lens:
            #since ex_lens and ey_lens are given in theta and phi coordinates, the have to be transformed to cartesian in order to be ploted, hence the name of this function
            plot_in_cartesian(ex_lens,ey_lens,h*10**-6,alpha,f,figure_name)
        else:
            #calculate field inciding on the lens by fraunhofer's difraction formula
            #N_rho and N_phi are the number of divisions for fraunhoffer's integral by the trapecium's method
            N_rho=400
            N_phi=400                
            ex_lens,ey_lens,I_cartesian,Ex_cartesian,Ey_cartesian=custom_mask_objective_field(gamma,beta,resolution_theta,resolution_phi,N_rho,N_phi,alpha,maskfunction,R,L,I0,wavelength/n,w0*10**-6,figure_name,plot=True)

        #calculate field at the focal plane:
        if ns=='Not implemented' and ds=='Not implemented':#no interphase
            ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=custom_mask_focus_field_XZ_XY(ex_lens,ey_lens,alpha,h,wavelength/n,z_field_of_view,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,field_of_view)
        else:
            ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=interphase_custom_mask_focus_field_XZ_XY(self.ns,self.ds,ex_lens,ey_lens,alpha,h,wavelength,self.zint,z_field_of_view,resolution_z,zp0,resolution_focus,resolution_theta,resolution_phi,field_of_view,x0=0,y0=0,z0=0,plot_plane='X')
        self.amplitudes_xy=(ex_XY,ey_XY,ez_XY)#used to generate the save file
        self.Ifield_xz=np.abs(ex_XZ)**2+np.abs(ey_XZ)**2+np.abs(ez_XZ)**2#used to generate the save file
        self.Ifield_xy=np.abs(ex_XY)**2+np.abs(ey_XY)**2+np.abs(ez_XY)**2#used to generate the save file
        
        return ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY

    def simulate(self):
        self.counter+=1
        NA,n,h,f,w0,wavelength,gamma,beta,zp0,rsteps,zsteps,field_of_view,z_field_of_view,I0,L,R,figure_name=self.get_parameters()
        
        #used to avoid overwriting previous figures which have the same name:
        if figure_name==self.default_file_name:
            figure_name+=' '+str(self.counter)
        if figure_name==self.previous_figure_name:
            figure_name+=' '+str(self.counter)
        self.previous_figure_name=figure_name        
        self.figure_name=figure_name #used as name for saving texts 
        
        alpha=np.arcsin(NA / n)
        selected=self.comboBox.currentIndex()
        try:                             
            propagation=self.radioButton.isChecked() #then no need to calculate the field at the entrance of the lens
            if selected==0: #VPP mask
                #calculate field at the focal plane:
                ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=self.VPP(propagation,NA,n,h,f,w0,wavelength,gamma,beta,zp0,rsteps,zsteps,field_of_view,z_field_of_view,I0,L,R,figure_name,self.ns,self.ds)                
                #plot the fields at the focus:
                plot_XZ_XY(ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY,field_of_view,z_field_of_view,figure_name) #ex, ey and ez have tha values of the field on the XY plane
                    
            if selected==1: #No mask (gaussian beam)
                if propagation==True:
                    print('Propagation for field at objective only for VPP and used defined masks')
                else:
                    #calculate field at the focal plane:
                    ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=self.No_mask(NA,n,h,f,w0,wavelength,gamma,beta,zp0,rsteps,zsteps,field_of_view,z_field_of_view,I0,L,R,figure_name,self.ns,self.ds)
                    #plot the fields at the focus:
                    plot_XZ_XY(ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY,field_of_view,z_field_of_view,figure_name) #ex, ey and ez have tha values of the field on the XY plane
            
            if selected==2: #Custom mask
                #calculate field at the focal plane:
                ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=self.Custom(propagation,NA,n,h,f,w0,wavelength,gamma,beta,zp0,rsteps,zsteps,field_of_view,z_field_of_view,I0,L,R,figure_name,self.ns,self.ds)
                #plot the fields at the focus:
                plot_XZ_XY(ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY,field_of_view,z_field_of_view,figure_name) #ex, ey and ez have tha values of the field on the XY plane

        except:
            print("Unexpected error:", sys.exc_info())

    def closeEvent(self, event, *args, **kwargs):
        print('PyFocus is closed')        
        super().closeEvent(*args, **kwargs)

        
        
if __name__ == '__main__':        
    gui=PyFocus()
    gui.show()
