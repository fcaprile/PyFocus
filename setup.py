from distutils.core import setup
setup(
  name = 'PyCustomFocus',         
  packages = ['PyFocus'],   
  version = '1.2',      
  license='MIT',        
  description = 'Full vectorial calculation of focused electromagnetic fields moduled by a custom phase mask',   
  author = 'Caprile Fernando',                  
  author_email = 'fcaprile@gmail.com',      
  url = 'https://github.com/fcaprile/PyFocus',  
  download_url = 'https://github.com/fcaprile/PyFocus/archive/refs/tags/1.2.tar.gz',    
  keywords = ['User interface', 'Custom phase mask'],   # Keywords that define your package best
  install_requires=[            
],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)