'''

 This script is used to create an artificial flow field to examine the response
 of the staggered-PRT correction success.

 '''

import numpy as np
import matplotlib.pyplot as plt
import sys
import ctables
from scipy.interpolate import UnivariateSpline
from testPRFCorrect import prtCorrectNew

def returnPRF(prfHigh,prfRatio,wLen):
    '''
    +++ Description +++

    Returns the low PRF and nyquists and the high and low PRFs.

    +++ Input +++

    prfHigh:    The PRF (Hz) for the short pulse.

    prfRatio:   The ratio of the low to high PRF (e.g., 2/3, 3/4, or 4/5).

    wLen:       The wavelength of the hypothetical radar (m).

    '''
    
    prfLow = prfHigh * prfRatio

    nyqHigh = wLen * prfHigh / 4.
    nyqLow = wLen * prfLow / 4.

    return prfLow,nyqHigh,nyqLow

def staggeredPRT(dopplerTrue,prfHigh,prfLow,wLen):

    '''
    +++ Description +++

    Function to compute Doppler velocities based on the staggered PRT technique
    from a "true" Doppler velocity field as input. The true wind field is aliased into
    two Doppler velocity intervals based on the low and high PRFs at a constant wavelength.

    +++ Input +++

    dopplerTrue: A hypothetical PPI (m/s) of "true" Doppler velocities based on a some known 
                 wind field.
    
    prfHigh:    The PRF (Hz) for the short pulse.

    prfLow:     The PRF (Hz) for the long pulse.

    wLen:       The wavelength of the hypothetical radar (m).

    +++ Returns +++

    VHigh:      The Doppler velocity field as sampled by the high PRF (short pulse).

    VLow:       The Doppler velocity field as sampled by the low PRF (long pulse).

    vavg:       The output staggered PRT velocities.
    '''
    
    #Calculate the phases across the low and high PRF intervals.
    phaseLow = (4 * np.pi * dopplerTrue) / (prfLow * wLen)
    phaseHigh = (4 * np.pi * dopplerTrue) / (prfHigh * wLen)

    #Adjust to bring within -pi to pi interval.
    for n in np.arange(1,11,2)[::-1]:
        phaseLow[phaseLow>=(n*np.pi)] = phaseLow[phaseLow>=(n*np.pi)] - ((n+1) * np.pi)
        phaseLow[phaseLow<=(-n*np.pi)] = phaseLow[phaseLow<=(-n*np.pi)] + ((n+1) * np.pi)
        phaseHigh[phaseHigh>=(n*np.pi)] = phaseHigh[phaseHigh>=(n*np.pi)] - ((n+1) * np.pi)
        phaseHigh[phaseHigh<=(-n*np.pi)] = phaseHigh[phaseHigh<=(-n*np.pi)] + ((n+1) * np.pi)

    #Add some noise to make the obs more realistic.
    phaseLow = phaseLow + np.random.normal(0,np.pi/12,phaseLow.shape)
    phaseHigh = phaseHigh + np.random.normal(0,np.pi/12,phaseHigh.shape)    

    #Get Nyquists
    prfLow,nyqHigh,nyqLow = returnPRF(prfHigh,prfRatio,wLen)

    #Compue the phase difference between high and low PRF samples.
    phaseDiff = phaseHigh - phaseLow

    #Use the smallest difference in the phase angles.
    phaseDiff[phaseDiff>np.pi] = phaseDiff[phaseDiff>np.pi] - 2*np.pi
    phaseDiff[phaseDiff<-np.pi] = phaseDiff[phaseDiff<-np.pi] + 2*np.pi

    #Compute the corrected Doppler velocity (high noise). This is Vc in papers.
    VCHigh = wLen * (phaseDiff) / (4 * np.pi * ((1./prfHigh) - (1./prfLow)))

    #Compute the Doppler velocity for high PRF interval.
    VHigh = phaseHigh * wLen * prfHigh / (4 * np.pi)

    #Compute possibe solutions for n = -10 to n = 10.
    vsolutions = []
    for n in range(-10,11,1):
        vsolutions.append(VCHigh - VHigh - 2*n*nyqHigh)
    diff = np.abs(np.array(vsolutions)) - nyqHigh

    #Only interested in solutions for |Vc-V-2nVnyq| < Vnyq
    vsolutions = np.ma.masked_where(np.abs(np.array(vsolutions)) > nyqHigh,diff).filled(fill_value = np.nan)

    #Retain best solution of n. Index is 10 offset, hence the -10
    nHigh = np.nanargmin(vsolutions,axis=0) - 10.

    #Calculate final based on retained n.
    vHighEstimate = VHigh + (2 * nHigh * nyqHigh)

    #Repeat for low PRF.
    phaseDiff = phaseLow - phaseHigh

    phaseDiff[phaseDiff>np.pi] = phaseDiff[phaseDiff>np.pi] - 2*np.pi
    phaseDiff[phaseDiff<-np.pi] = phaseDiff[phaseDiff<-np.pi] + 2*np.pi

    VCLow = wLen * (phaseDiff) / (4 * np.pi * ((1./prfLow) - (1./prfHigh)))

    VLow = phaseLow * wLen * prfLow / (4 * np.pi)

    vsolutions = []
    for n in range(-10,11,1):
        vsolutions.append(VCLow - VLow - 2*n*nyqLow)
    diff = np.abs(np.array(vsolutions)) - nyqLow
    vsolutions = np.ma.masked_where(np.abs(np.array(vsolutions)) > nyqLow,diff).filled(fill_value = np.nan)
    nLow = np.nanargmin(vsolutions,axis=0) - 10.

    vLowEstimate = VLow + (2 * nLow * nyqLow)

    #Take average of both estimates to get final value.
    vavg = (vHighEstimate + vLowEstimate) / 2.

    return VHigh,VLow,vavg

def windProfile():

    height = [0,200,500,750,1000,1500,2000,3000,4000]
    windU = [-5,-3,-1,-1,-1,-1,-1,-1,-1]
    windV = [0,15,45 ,45,42,41,38,37,35]

    return height,windU,windV


def makeGrid(prfHigh,pulseLength,rangeLim = 100,elevation=1.0):
    '''
    +++ Description +++

    Returns an artifical PPI of Doppler velocities based on windMax.

    Function assumes 1 degree PPI resolution.

    +++ Input +++

    prfHigh:    The high PRF in Hz.

    rangeLim:   The range limit of the PPI in km.

    elevation:  The hypothetical elevation angle for PPI (degrees).

    +++ Returns +++

    x:          The x-distance (km) from the radar.

    y:          The y-distance (km) from the radar.

    radius:     The radial locations of gates (km).

    azimuths:   The azimuthal location of gates from north (degrees).

    dopplerTrue: The true Doppler velocity field based on the full u and v wind.

    '''
    
    sol = 3e8 #speed of light in m/s

    rangeUnambiguous = sol / (prfHigh * 2. * 1000.)
    
    if rangeLim > rangeUnambiguous: rangeLim = rangeUnambiguous

    binSpacing = 0.5 * (sol * pulseLength) / 1000.

    rgrid,azgrid = np.meshgrid(np.arange(0,rangeLim,binSpacing),range(0,360,1),indexing='ij')
    
    deg_to_rad = np.pi / 180.
    a_e = (8494.66667)*1000

    a = 1000 * rgrid**2 + a_e**2 + (2.0 * 1000 * rgrid * a_e * np.sin(elevation * deg_to_rad))
    h =((a**0.5) - a_e) + 2

    x,y = rgrid * np.sin(np.radians(azgrid)),rgrid * np.cos(np.radians(azgrid))

    #u,v = windMax * np.log((h-100.) / 0.1),windMax * np.log((h-100.) / 0.1)
    #u,v, = windMax,windMax*np.ones(x.shape)*(rgrid/rangeLim)
    height,uData,vData = windProfile()

    ufunc = UnivariateSpline(height,uData)
    vfunc = UnivariateSpline(height,vData)

    u = ufunc(h.flatten()).reshape(x.shape)
    v = vfunc(h.flatten()).reshape(y.shape)

    dopplerTrue = (u * np.sin(np.radians(azgrid))) + (v * np.cos(np.radians(azgrid)))

    return x,y,np.array(np.arange(0,rangeLim,binSpacing)),np.array(range(0,360,1)),dopplerTrue

#### Run program ###
prfHigh = 1400 #Hz
prfRatio = (4./5.)
wLen = 0.05 #m
windMax = 45 #m/s
pulseLength = 0.5 #microsecond

pulseLength = pulseLength * 1e-6

dutyCycle = pulseLength * prfHigh

if (1000*dutyCycle) > 0.95:
    print("You must select a pulse length/high PRF that is < 0.95 of the duty cycle.")
    print("Your current duty cycle is %.2f"%dutyCycle)
    sys.exit()

prfLow,nyqHigh,nyqLow = returnPRF(prfHigh,prfRatio,wLen)

x,y,radius,azimuth,dopplerTrue = makeGrid(prfHigh,pulseLength,rangeLim = 100,elevation=1.0)

VHigh,VLow,vavg = staggeredPRT(dopplerTrue,prfHigh,prfLow,wLen)

diff = dopplerTrue - vavg

print(nyqHigh+nyqLow,nyqHigh,nyqLow)

velCorrected = prtCorrectNew(vavg.copy().T,radius*1000,azimuth,nyqL=nyqLow,nyqH=nyqHigh,fnyq=nyqHigh+nyqLow).T

#Compute stats
diffCorrected = dopplerTrue - velCorrected

errors = np.where(np.abs(diff)>5)[0].shape[0]
remaining = np.where(np.abs(diffCorrected)>5)[0].shape[0]

fig = plt.figure(1)
fig.set_size_inches(5,15)
#for n,vel,title in \
#    zip(range(221,225,1),[dopplerTrue,velCorrected,VHigh,VLow],['True','Estimated','High PRF','Low PRF']):
for n,vel,title in \
     zip(range(311,314,1),[dopplerTrue,vavg,velCorrected],['True','Estimated','Corrected']):
    
    ax = fig.add_subplot(n)

    a = ax.pcolormesh(x,y,vel,vmin=-windMax,vmax=windMax,cmap=ctables.BuDRd18)

    ax.set_title(title)

fig.subplots_adjust(left=0.07,bottom=0.15,top=0.96,right=0.97)
plt.colorbar(a,orientation='horizontal',cax =fig.add_axes([0.07,0.05,0.9,0.03]))

print("%d of %d errors estimated remaining..."%(remaining,errors))

fig.savefig('testCase.png',dpi=300)
