import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "PyCustomFocus",
    version = "3.3.2",
    author = "Caprile Fernando",
    author_email = "fcaprile@gmail.com",
    description = 'Full vectorial calculation of focused electromagnetic fields moduled by a custom phase mask',   
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = 'https://github.com/fcaprile/PyFocus',  
    classifiers=[
      'Development Status :: 3 - Alpha',      
      'Intended Audience :: Developers',      
      'Topic :: Software Development :: Build Tools',
      'License :: OSI Approved :: MIT License',   
      'Programming Language :: Python :: 3.9'
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.8"
)