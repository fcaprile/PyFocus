from distutils.core import setup
setup(
  name = 'PyCustomFocus',         
  packages = ['PyFocus'],   
  version = '0.3.7',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Full vectorial calculation of focused electromagnetic fields moduled by a custom phase mask',   # Give a short description about your library
  author = 'Caprile Fernando',                   # Type in your name
  author_email = 'fcaprile@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/fcaprile/PyFocus',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/fcaprile/PyFocus/archive/refs/tags/0.3.7.tar.gz',    # I explain this later on
  keywords = ['User interface', 'Custom phase mask'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
    'config',
    'numpy',
    'scipy',
    'qdarkstyle',
    'configparser',
],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)