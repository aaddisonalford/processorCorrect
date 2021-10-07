"""

Contains the staggered PRT and dual PRF correction method.

Both methods can be called through the errorCorrect function.

Requires scipy, numpy, datetime, and sys.

"""

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import numpy as np
from scipy.stats import norm
from datetime import datetime
import sys


def errorCorrect(radar,velField = 'VT',fnyq = 0,nyqL=0,nyqH=0,method = 'staggered'):

    '''
    +++ Description +++

    This function corrects errors related to staggered-PRT processing.

    +++ Input +++

    radar:  The radar object from Py-ART.

    velField:   The velocity field name (should already be dealiased).

    fnyq:   This value is the sum of the low and high PRF nyquists.

    nyqL:   The low PRF nyquist in m/s.

    nyqH:   The high PRF nyquist in m/s.

    method: Either 'staggered' or 'dual' for staggered PRT correction or dual PRF
            correction respectively.

    +++ Returns +++

    The radar object with the velField corrected for staggered-PRT errors.

    '''

    # Set method and filter lengths.
    if method == 'staggered':
        staggeredPRT,dualPRF = True,False
        filters = zip([11,21,5,51,71,5],[5,9,5,21,71,5])

    elif method == 'dual':
        dualPRF,staggeredPRT = True,False
        filters = zip([71,11,21,5,51,71,5],[71,5,9,5,21,71,5])

    else:
        print("Method is not recognized.")
        print("Please specify 'staggered' or 'dual'.")
        sys.exit()

    timeNow = datetime.now()

    pointsCaught = 0
    for radFilter,azFilter in filters:
        for sweep_slice in radar.iter_slice():
            
            #### Radial ####
            
            #Copy velocity field and mask.
            vel = radar.fields[velField]['data'][sweep_slice].copy()
            vel = np.ma.masked_outside(vel.copy(),-500,500)

            #Do radials first, so collect range information. km or m does not matter.
            r = radar.range['data']
            
            #Linear interpolation along radial.
            velInterp = np.empty(vel.shape)
            for ray,num in zip(vel,list(range(velInterp.shape[0]))):
                if ray.compressed().shape[0] > 5:
                    velInterp[num] = np.interp(r,r[ray.mask == False],ray.compressed())
                else: velInterp[num] = ray.filled(fill_value = np.nan)
            
            #Smooth interpolated field.
            velSmoothRad = savgol_filter(velInterp, radFilter, 3, mode='interp',axis=1)

            #Compute radial difference of original velocities (vel) from smoothed field.
            diffRad = vel.filled(fill_value=np.nan) - velSmoothRad
            
            #### Azimuthal ####

            #Copy velocity field again and transpose.
            vel = radar.fields[velField]['data'][sweep_slice].copy().T
            vel = np.ma.masked_outside(vel.copy(),-500,500)

            #This time, r is the azimuth field.
            r = radar.azimuth['data'][sweep_slice]
            
            #Linearly interpolate the azimuths.
            velInterp = np.empty(vel.shape)
            velInterp.shape[0]
            for ray,num in zip(vel,list(range(velInterp.shape[0]))):
                if ray.compressed().shape[0] > 5:
                    rIn = r[ray.mask == False]
                    rayIn = ray.compressed()
                    indices = rIn.argsort()
                    
                    velInterp[num] = np.interp(r,rIn[indices],rayIn[indices])
                else: velInterp[num] = ray.filled(fill_value = np.nan)
            
            #Smooth the linearly interpolated azimuthal field.
            velSmoothAz = savgol_filter(velInterp, azFilter, 3, mode='interp',axis=1).T

            #Compute the mean smoothed velocity field.
            velSmoothMean = np.nanmean(np.array([velSmoothAz,velSmoothRad]),axis=0)
            
            #Compute azimuthal difference of original velocities (vel) from smoothed field.
            diffAz = vel.T.filled(fill_value=np.nan) - velSmoothAz

            #Compute MEAN difference of original velocities (vel) from smoothed field.
            diffMean = vel.T.filled(fill_value=np.nan) - velSmoothMean
            
            #Copy vel field for testing.
            velNew = vel.copy().T

            #Compute the standard deviation of the azimuthal difference field.
            mu,stdAz = norm.fit(np.ma.masked_invalid(diffAz).compressed())

            #Fill NaNs where azimuthal difference is < 3*(standard deviation).
            diffAz = np.ma.masked_where(np.abs(diffAz) < 3*stdAz,diffAz).filled(fill_value = np.nan)
            
            #Compute the standard deviation of the radial difference field.
            mu,stdRad = norm.fit(np.ma.masked_invalid(diffRad).compressed())
            
            #Fill NaNs where radial difference is < 3*(standard deviation).
            diffRad = np.ma.masked_where(np.abs(diffRad) < 3*stdRad,diffRad).filled(fill_value = np.nan)
            
            #Initialize arrays for staggered PRT correction.
            #Also set "bound" which is the width of the search for errors.
            #fnyq = VNyqHigh + VNyqLow
            if staggeredPRT:
                possibleSolutions = np.empty((10,velNew.shape[0],velNew.shape[1]))
                differences = np.empty((10,velNew.shape[0],velNew.shape[1]))
                bound = fnyq

            #Initialize arrays for dual PRF correction.
            if dualPRF:
                possibleSolutions = np.empty((7,velNew.shape[0],velNew.shape[1]))
                differences = np.empty((7,velNew.shape[0],velNew.shape[1]))

            #First solution is the original velocity field.
            possibleSolutions[0,:] = velNew.copy()
            differences[0,:] = velNew.copy() - velSmoothMean
            
            #Initialize count = 1 because of prior entry of original velocity field.
            count = 1

            #Staggered PRT Correction for all solutions.
            if staggeredPRT:
                #Loop over n1, n2 = 1,2,3.
                for n1 in [1,2,3]:
                    for n2 in [1,2,3]:
                        #Error is sum of n1(VNLow) + n2(VNHigh)
                        nyq = n1*nyqL + n2*nyqH                  

                        #Copy original/partially corrected field.
                        velPossible = velNew.copy()

                        #Search for positive errors where azimuthal AND radial differences are greater than
                        #3 standard deviations and meet error criteria.
                        positiveIndices = np.where((diffAz != np.nan) & (diffRad != np.nan) & \
                            (diffMean > nyq - bound) & (diffMean < nyq + bound))

                        #Correct positive errors.
                        velPossible[positiveIndices] = velPossible[positiveIndices] - nyq
                        
                        #As above, but for negative errors.
                        negativeIndices = np.where((diffAz != np.nan) & (diffRad != np.nan) & \
                            (diffMean < -1*(nyq - bound)) & (diffMean > -1*(nyq + bound)))

                        #Correct negative errors.
                        velPossible[negativeIndices] = velPossible[negativeIndices] + nyq

                        #Save solution for n1, n2.
                        possibleSolutions[count,:] = velPossible

                        #Compute solution difference from smoothed, mean field.
                        differences[count,:] = velPossible-velSmoothMean
                        
                        #Next!
                        count+=1

            #Correction for dual PRF errors.
            if dualPRF:
                #Loop over n = 1,2,3
                for n1 in [1,2,3]:
                    #Loop over both nyquists with bound being 2*n*nyq.
                    for nyq,bound in zip([2*n1*nyqL,2*n1*nyqH],[2*n1*nyqL,2*n1*nyqH]): 
                        
                        #Copy original/partially corrected Doppler velocities.
                        velPossible = velNew.copy()

                        #As in staggered PRT correction method, check that gates meet error criteria for
                        #positive errors.
                        positiveIndices = np.where((diffAz != np.nan) & (diffRad != np.nan) & \
                            (diffMean > nyq - bound) & (diffMean < nyq + bound))

                        #Correct positive errors.
                        velPossible[positiveIndices] = velPossible[positiveIndices] - nyq
                        
                        #Check for error criteria for negative errors.
                        negativeIndices = np.where((diffAz != np.nan) & (diffRad != np.nan) & \
                            (diffMean < -1*(nyq - bound)) & (diffMean > -1*(nyq + bound)))

                        #Correct negative errors.
                        velPossible[negativeIndices] = velPossible[negativeIndices] + nyq

                        #Save possible solution for n1 and nyquist.
                        possibleSolutions[count,:] = velPossible

                        #Compute difference of solution relative to smoothed mean field.
                        differences[count,:] = velPossible-velSmoothMean

                        #On to the next one.
                        count+=1

            #Set NaNs in the differences to 0.
            differences = np.abs(np.ma.masked_invalid(differences).filled(fill_value=0.))

            #Get azimuth, range indices.
            azimuths,ranges = np.meshgrid(range(velSmoothMean.shape[0]),range(velSmoothMean.shape[1]),indexing='ij')
            
            #Create tuple of indices where difference field is minimized.
            indices = tuple([np.nanargmin(differences,axis=0).flatten(),azimuths.flatten(),ranges.flatten()])

            #Calculate the total corrections made.
            pointsCaught = pointsCaught + np.where(np.nanargmin(differences,axis=0).flatten()!=0)[0].shape[0]

            #Save final solution and reshape.
            finalSolution = possibleSolutions[indices].reshape(velSmoothMean.shape)

            #Update the radar object.                
            radar.fields[velField]['data'][sweep_slice] = np.ma.masked_where(radar.fields[velField]['data'][sweep_slice].mask,finalSolution)
            
    print("TOTAL TIME %.2f"%((datetime.now()-timeNow).seconds))    
    print("TOTAL POINTS CAUGHT ",pointsCaught)
    
    #Return the radar object.
    return radar

def getNyq(radar,radarName):
    nyq = round(radar.get_nyquist_vel(0), 3)
    freq = radar.instrument_parameters['frequency']['data'].copy()[0]
    c = 2.9989e8
    wLen = c/freq
    
    if 'DOW' in radarName:
        vLow = wLen/(radar.instrument_parameters['prt']['data'].copy()[0]*4)
        if vLow < 1.:
            tmp = vLow *10**3
            if tmp < 1. or tmp > 100.:
                print('You need to check the PRT for DOW.')
                print('Proceeding with a PRT of %.2f us.'\
                    %radar.instrument_parameters['prt']['data'].copy()[0])
                tmp = vLow
            vLow = tmp
    elif 'NOXP' in radarName:
        vLow = wLen/(radar.instrument_parameters['prt']['data'].copy()[0] \
            * 10**-3 *4)
        if vLow < 1.:
            tmp = vLow *10**3
            if tmp < 1. or tmp > 100.:
                print('You need to check the PRT for NOXP.')
                print('Proceeding with a PRT of %.2f us.'\
                    %radar.instrument_parameters['prt']['data'].copy()[0])
                tmp = vLow
            vLow = tmp
    else:
        vLow = wLen/(radar.instrument_parameters['prt']['data'].copy()[0]*4)
        if vLow > nyq:
            vLow = vLow /3.
        if vLow > nyq:
            print('Folding Nyquist cannot be determined.')
        if vLow < 1.:
            tmp = vLow *10**3
            if tmp < 1. or tmp > 100.:
                print('You need to check the PRT for NOXP.')
                print('Proceeding with a PRT of %.2f us.'\
                    %radar.instrument_parameters['prt']['data'].copy()[0])
                tmp = vLow
            vLow = tmp
    ratioLow = nyq/vLow
    
    ratioLow = int(np.round(ratioLow,0))
    ratioHigh = ratioLow + 1

    vHigh = nyq/ratioHigh
    
    foldNyq = vHigh + vLow

    return vLow,vHigh,foldNyq