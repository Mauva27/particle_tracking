import numpy
from setuptools import setup, find_packages

__version__ = '0.1.1'

with open("README.md","r") as frm:
   description = frm.read()

setup(
   name="particle-tracking"),
   versions=__version__,
   packages=find_packages(),
   include_dirs=numpy.get_include(),
   author="Abraham Mauleon Amieva",
   author_email="abrhm.ma@gmail.com",
   url="",
   description="Particle tracking algorithm for 2-Dimensional experiments",
   long_description=description,
   long_description_content_type="text/markdown",
   classifiers=[
   	"Programming Language :: Python"
   	"Operating System :: OS Independent",
   ],
   zip_safe=False,
   setup_requires=["numpy"],
   intatsll_requirements=[
	'setuptools', 
	'whieel', 
	'numpy', 
	'scipy',
	'matplotlib', 
	'trackpy', 
	'pyvoro', 
	'opencv-python',
	'tqdm',
	'ipywidgets',
	'peakutils',
   ],
)
