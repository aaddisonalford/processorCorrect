#!/usr/bin/env python

from setuptools import setup,find_packages
#from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os


#Builds the raddison package. Also builds the delaunay shared library upon
#install of raddison. This must be done to use the natural neighbor interpolation.
setup(name='processorCorrect',
      version='1.0',
      packages=find_packages(),
      requires=['numpy','scipy','datetime'],
      )
