# ProcessorCorrect

# Description
This software is used to correct errors related to weather radar Nyquist extension techniques. Staggered Pulse Repetition Time (PRT) is a technique employed by weather radar wherein the PRF/PRT is changed from pule-pair to pulse-pair. Using the Doppler velocity information from each PRF/PRT time series, the individual pulse-pairs can use the corresponding sample at the other PRF/PRT to dealias the Doppler velocities into a larger Nyquist interval. Due to enhanced phase noise from combining the two pulse-pairs, errors (here referred to as "processor errors") are often found in regions of high shear, low signal-to-noise ratio, and high spectrum width.

Similarly, dual PRF processing similarly changes PRF/PRT but from azimuth to azimuth. A former azimuth is assumed to be representative of the Doppler velocities in next (i.e., actual flow is assumed to be approximately constant from azimuth to azimuth) and is used to dealias the current azimuth. Errors can be found, particularly in regions of strong shear.

This software can be used to correct such processor errors. The Python ARM Radar Toolkit is used to read in a particular radar file and pass the pre-dealiased Doppler velocity field to processorCorrect.

# Citing
If you use this software to prepare a publicaiton, please cite:

Alford, A. A., M. I. Biggerstaff, C. L. Ziegler, D. P. Jorgensen, and G. D. Carrie (2022): A method for correcting staggered pulse repetition time (PRT) and dual pulse repetition frequency (PRF) processor erorrs in research radar datasets. In review in the Journal of Atmospheric and Oceanic Technology. DOI information to follow.

# Install

Clone or download this repository.

cd processorCorrect-main

python setup.py build

python setup.py install

# Use

import pyart

import processorCorrect

radar = pyart.io.read('/path/to/radar/file')

radar = processorCorrect.errorCorrect(radar,velField = 'VT',fnyq = 40,nyqL=16,nyqH=24,method = 'staggered')

# Dependencies

pyart

numpy

scipy

datetime
