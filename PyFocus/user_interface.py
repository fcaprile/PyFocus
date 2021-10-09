#usual packages
import numpy as np
from matplotlib import pyplot as plt
import time
import configparser
import config
from PyQt5.QtWidgets import QFileDialog
from qtpy import QtWidgets
from pyqtgraph.Qt import QtGui
import qdarkstyle

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

#from Pyqt5 user interface packages
from PyFocus.front_end_ui import Ui_MainWindow
# from PyFocus.mask_selection import Ui_Dialog
from PyFocus.mask_selection import Ui_MaskWindow

#custom made integration functions
from PyFocus.plot import plot_XZ_XY
from PyFocus.sim import VP, no_mask, custom

'''
User interface class that allows seting up parameters, running simulations and saving the obtained fields as .txt files
'''

class Mask_window(QtGui.QMainWindow,Ui_MaskWindow):
    '''
    Extra window for the selection of a custom phase mask
    '''
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.ui = Ui_MaskWindow()
        self.ui.setupUi(self)


class UI(QtGui.QMainWindow,Ui_MainWindow):
    '''
    User Interface class for simulatons of PyFocus
    '''
    def __init__(self, *args, **kwargs):
        print('PyFocus is running')    
        time.sleep(0.2)#so that tqdm can write
        #if running locally, uncomment these 2 lines:
        self.app = QtGui.QApplication.instance()#if running from spyder
        if self.app is None:
            self.app = QtGui.QApplication(sys.argv)#if running from console
        self.app.setStyleSheet(qdarkstyle.load_stylesheet())#set a dark style
        
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        #Main functions:
        self.counter=0# counts the numer of simulations realized in this sesion, to avoid overwriting save files with the same name
        self.setupUi(self)
        self.pushButton.clicked.connect(self.save_parameters)
        self.pushButton_2.clicked.connect(self.save_intensity)
        self.pushButton_3.clicked.connect(self.simulate)
        self.pushButton_4.clicked.connect(self.save_amplitudes)        
        self.pushButton_5.clicked.connect(self.selectsavefolder)   
        self.pushButton_6.clicked.connect(self.clear_plots)   
        self.pushButton_7.clicked.connect(self.open_custom_mask_window)   
        self.pushButton_8.clicked.connect(self.load_field_from_txt)   
        self.radioButton.toggled.connect(self.lineEdit_19.setEnabled)#enable modifying parameters for simulation of the field inciding on the lens
        self.radioButton.toggled.connect(self.lineEdit_22.setEnabled)
        self.radioButton_2.toggled.connect(self.lineEdit_23.setEnabled)#enable modifying parameters for an interface
        self.radioButton_2.toggled.connect(self.lineEdit_24.setEnabled)
        self.radioButton_2.toggled.connect(self.lineEdit_25.setEnabled)
        self.radioButton_2.toggled.connect(self.lineEdit_2.setDisabled)
        
        #Functions dedicated to the saved files's names
        self.lineEdit_21.textEdited.connect(self.change_saving_name)
        self.comboBox.currentTextChanged.connect(self.change_default_name_and_open_dialog)
        self.modified_saving_name=False #Used later to avoid overwriting the changes in the name of the figure made by the user
        self.default_file_name='VP mask simulation' #default name for simulations if nothing is changed
        self.previous_figure_name=''#used to aoid overwriting previous figures which have the same name
        config.y=True
        
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
        h=float(self.lineEdit_18.text()) #radius of aperture converted to nm
        if self.comboBox_2.currentIndex()==0:#Gaussian beam
            w0=float(self.lineEdit_16.text()) #incident gaussian beam's radius, converted to nm
        else:
            w0=10000#uniform field
        wavelength=float(self.lineEdit_3.text())#wavelength in vacuum
        I0=float(self.lineEdit_15.text())#maximum intensity for the incident beam (|A|^2)
        zp0=float(self.lineEdit_12.text())#axial distance in wich to calculate the XY plane (z=cte=zp0 is the plane)
        gamma=float(self.lineEdit_11.text()) #arctan(ey/ex)
        beta=float(self.lineEdit_13.text()) #Delta phase betheen ey and ey, gamma=45 and beta=90 give right circular poalarizacion and a donut, gamma=45 and beta=-90 give left polarization and a maximum of intensity in the focused field
        x_steps=float(self.lineEdit_5.text())#radial pixel size
        z_steps=float(self.lineEdit_4.text())#axial pixel size
        x_range=float(self.lineEdit_17.text())#radial distance for the simulation
        if self.radioButton.isChecked()==True:#L and R are parameters for calculating the inciding beam with fraunhoffer's formula, if the propagation is depreciated then this parameters ar saved as ''not implemented''
            L=float(self.lineEdit_19.text())
            R=float(self.lineEdit_22.text())
        else:
            L='Not used'
            R='Not used'
        z_range=float(self.lineEdit_20.text())#axial distance for the simulation
        figure_name=str(self.lineEdit_21.text())#name for the ploted figures and the save files
        if figure_name=='': #to always have a figure name
            figure_name='Simulation'
        if self.radioButton_2.isChecked()==True:#if the interface simulation is not checked, this parameters ar saved as ''not implemented''
            try:
                z_int=float(self.lineEdit_25.text())#axial distance of the interface (negatie means before the focal plane)
                n=np.array([complex(a) for a in self.lineEdit_23.text().split(',')])#array containing the refraction indexes of all mediums
                ds=[]#array cointaining the thickness of all mediums
                ds.append(np.inf)
                if not self.lineEdit_24.text()=='':
                    for d in [float(s) for s in self.lineEdit_24.text().split(',')]:
                        ds.append(d)
                ds.append(np.inf)
                ds=np.array(ds)
            except:
                print('Wrong format, explample of correct format: ns=1.5,1.4,1.4+1.4j,1.2j,1.33, ds=5,10,2',sys.exc_info())
            if not len(ds)==len(n):
                print('The refraction index array or the thickness array are missing a parameter, check that there are 2 parameters less for thickness than for refraction index')
                raise ValueError('The refraction index array or the thickness array are missing a parameter, check that there are 2 parameters less for thickness than for refraction index')
        else:
            z_int='Not used'
            ds='Not used'
        self.parameters=np.array((NA,n,h,w0,wavelength,gamma,beta,zp0,x_steps,z_steps,x_range,z_range,I0,L,R,ds,z_int,figure_name), dtype=object)

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
            
            radius=np.copy(self.parameters[3])
            if radius==10000:#in case of uniform field
                radius='Uniform field'
            
            config = configparser.ConfigParser()
            # #check if an interface as simualted:
            # if len(self.parameters[1])==1:#no interface

            config['Parameters'] = {
            'NA': self.parameters[0],
            'Aperture radius (mm)':self.parameters[2],
            # 'focal distance (mm)': self.parameters[3],#removed since it is given by sine's law
            'incident beam radius (mm)': radius,
            'wavelength at vacuum (nm)': self.parameters[4],
            'gamma, arctan(ey ex) (degrees)': self.parameters[5],
            'beta, Delta phase (degrees)': self.parameters[6],
            'Axial distance from focus for the XY plane (nm)': self.parameters[7],
            'Radial pixel size (nm)': self.parameters[8],
            'Axial pixel size (nm)': self.parameters[9],
            'Radial field of view (nm)': self.parameters[10],
            'Axial field of view (nm)': self.parameters[11],
            'Laser intensity (mW/cm^2)': self.parameters[12],
            'refraction indices': self.parameters[1],
            'interface layer thickness (nm)': self.parameters[15],
            'interface axial position (nm)': self.parameters[16],
            'Distance from phase plate to objective (mm)': self.parameters[13],
            'Phase mask radius (mm)': self.parameters[14]}

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
        
    def open_custom_mask_window(self):
        try:
            self.ui2 = Mask_window()
            self.ui2.show()
        except:
            print("Unexpected error:", sys.exc_info())
        
    def change_default_name_and_open_dialog(self):
        '''
        If a new mask is selected and the name has never been modified, then change the default simulation name
	Open the phase mask selection window if the "custom" option is selected 
        '''
        selected=self.comboBox.currentIndex()
        if self.modified_saving_name==False:
            if selected==0:#VP mask
                self.default_file_name='VP mask simulation'
                self.lineEdit_21.setText(self.default_file_name)
            if selected==1:#No mask (gaussian beam)
                self.default_file_name='Gaussian beam simulation'
                self.lineEdit_21.setText(self.default_file_name)
            if selected==2:#Custom mask
                try:
                    self.default_file_name='Custom mask simulation'
                    self.lineEdit_21.setText(self.default_file_name)
                    self.ui2 = Mask_window()
                    self.ui2.show()
                except:
                    print("Unexpected error:", sys.exc_info())
    
    def load_field_from_txt(self):
        try:
            config.x=np.loadtxt(QtWidgets.QFileDialog.getOpenFileName(None,'Select mask function File')[0],dtype=complex)
            config.y=False #internal parameter to set whether or the mask function is a function or a txt
            print('File loaded')
        except:
            print('File could not be loaded')

    
    def simulate(self):
        '''
        Simulate with the UI using the functions provided in "sim"
        '''
        try:      
            self.counter+=1
            self.get_parameters()
            x_range=self.parameters[10]
            z_range=self.parameters[11]
            figure_name=self.parameters[-1]
            #used to avoid overwriting previous figures which have the same name:
            if figure_name==self.default_file_name:
                figure_name+=' '+str(self.counter)
            if figure_name==self.previous_figure_name:
                figure_name+=' '+str(self.counter)
            self.previous_figure_name=figure_name        
            self.figure_name=figure_name #used as name for saving texts 
            self.parameters[-1]=self.figure_name
            selected=self.comboBox.currentIndex()
        
            propagation=self.radioButton.isChecked() #then no need to calculate the field at the entrance of the lens
            interface=self.radioButton_2.isChecked()
            if config.y==True: #internal variable that checks if the incident field is given by a txt, True means no
                if selected==0: #VP mask
                    #calculate field at the focal plane:
                    ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=VP(propagation,interface,*self.parameters)                
                        
                if selected==1: #No mask (gaussian beam)
                    #calculate field at the focal plane:
                    ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=no_mask(propagation,interface,*self.parameters)
            
                if selected==2: #Custom mask
                    try:
                        aux='self.custom_mask=lambda rho,phi,w0,f,k:'+config.x 
                        entrance_field=lambda rho,phi,w0,f,k:np.exp(-(rho/w0)**2)
                        exec(aux)
                        #calculate field at the focal plane:
                        ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=custom(entrance_field,self.custom_mask ,propagation,interface,*self.parameters)
                    except:
                        print('Please define a phase mask with the "Define mask" button')
                        return
            else:
                self.custom_mask=config.x
                entrance_field=lambda rho,phi,w0,f,k:1
                #calculate field at the focal plane:
                ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY=custom(entrance_field,self.custom_mask ,propagation,interface,*self.parameters)
                config.y=True
                print('File unloaded, if you want to repeat the simulation please reload the file')
            #plot the fields at the focus:
            plot_XZ_XY(ex_XZ,ey_XZ,ez_XZ,ex_XY,ey_XY,ez_XY,x_range,z_range,figure_name) #ex, ey and ez have tha values of the field on the XY plane
            
            #make internal variables to save the fields as txts
            self.amplitudes_xy=(ex_XY,ey_XY,ez_XY)
            self.Ifield_xy=np.abs(ex_XY)**2+np.abs(ey_XY)**2+np.abs(ez_XY)**2#used to generate the save file
            self.Ifield_xz=np.abs(ex_XZ)**2+np.abs(ey_XZ)**2+np.abs(ez_XZ)**2#used to generate the save file
        except:
            print("Unexpected error:", sys.exc_info())

    def closeEvent(self, event):
        '''
     	Close the user interface
     	'''
        print('PyFocus is closed')        
        event.accept()
        
        
if __name__ == '__main__':     
    # Dialog = QtWidgets.QDialog()
    # ui2 = Ui_Dialog()
    # ui2.setupUi(Dialog)
    gui=UI()
    gui.show()
