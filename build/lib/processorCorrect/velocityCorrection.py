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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def smoothVel(vel,spatial,filter):

    '''
    +++ Description +++

    This function takes the velocity field input (vel) and returns a smooted
    velocity field and the difference of the raw field from the smoothed.

    +++ Input +++

    vel:        The observed Doppler velocity field.

    spatial:    Either the radius or azimuth data associated with the shape of vel.

    filter:     The radial or azimuthal filter window.

    +++ Returns +++

    The smoothed velocity field and the difference from the observed velocity field.
    
    '''

    #Linear interpolation along radial.
    velInterp = np.empty(vel.shape)
    for ray,num in zip(vel,list(range(velInterp.shape[0]))):
        if ray.compressed().shape[0] > 5:
            spatialIn = spatial[ray.mask == False]
            rayIn = ray.compressed()
            indices = spatialIn.argsort()
            velInterp[num] = np.interp(spatial,spatialIn[indices],rayIn[indices])
        else: velInterp[num] = ray

    #Smooth interpolated field.
    velSmooth = savgol_filter(velInterp, filter, 3, mode='interp',axis=1)

    #Change -32768 to NaNs.
    velSmooth[velSmooth < -1000] = np.nan

    #Create difference field
    diff = vel.filled(fill_value=np.nan) - velSmooth

    return velSmooth,diff


def errorCorrect(radar,velField = 'VT',fnyq = 0,nyqL=0,nyqH=0,method = 'staggered',plotStats = False, name='figure'):

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

    plotStats: True to plot a histogram before and after correction. Default False.

    name:   If plotStats is True, this can be used as a name for the figure output. For example,
            you may want to utilize the date/time info for each PPI processed by this software.

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
            
            #Compute the smoothed velocity field over the radial dimension and the difference.
            velSmoothRad,diffRad = smoothVel(vel,radar.range['data'],radFilter)
            
            #### Azimuthal ####

            #Copy velocity field again and transpose.
            vel = radar.fields[velField]['data'][sweep_slice].copy().T
            vel = np.ma.masked_outside(vel.copy(),-500,500)
            
            velSmoothAz,diffAz = smoothVel(vel,radar.azimuth['data'][sweep_slice],azFilter)
            velSmoothAz,diffAz = velSmoothAz.T,diffAz.T

            #Compute the mean smoothed velocity field.
            velSmoothMean = np.nanmean(np.array([velSmoothAz,velSmoothRad]),axis=0)
         
            #Compute MEAN difference of original velocities (vel) from smoothed field.
            diffMean = vel.T.filled(fill_value=np.nan) - velSmoothMean 
            
            #Copy vel field for testing.
            velNew = vel.copy().T

            #Compute the standard deviation of the azimuthal difference field.
            mu,stdAz = norm.fit(np.ma.masked_invalid(diffAz).compressed())
            
            #Compute the standard deviation of the radial difference field.
            mu,stdRad = norm.fit(np.ma.masked_invalid(diffRad).compressed())

            #Selects what points to limit based on either the Nyquist velocity or
            #the standard deviation of the radial and azimuthal distributions.
            #If both the Nyquist velocities are less than both 3 standard deviations
            #the low Nyquist velocity is used instead.
            if (max([nyqL,nyqH])) < 3*stdRad and (max([nyqL,nyqH])) < 3*stdAz:
                radMask,azMask = min([nyqL,nyqH]),min([nyqL,nyqH])
            else:
                radMask,azMask = 3*stdRad,3*stdAz

            #Fill NaNs where azimuthal difference is < 3*(standard deviation).
            diffAz = np.ma.masked_where(np.abs(diffAz) < azMask,diffAz).filled(fill_value = np.nan)
            
            #Fill NaNs where radial difference is < 3*(standard deviation).
            diffRad = np.ma.masked_where(np.abs(diffRad) < radMask,diffRad).filled(fill_value = np.nan)
            
            #Plotting the difference field as a histogram.
            if pointsCaught == 0 and plotStats  == True:
                fig = plt.figure(1)
                fig.clf()
                gs1 = gridspec.GridSpec(1,1)
                gs1.update(wspace=0.2,hspace=0.2)
                fig.set_size_inches(5,5)

                ax = fig.add_subplot(gs1[0])

                a=ax.hist(diffMean.flatten(),range(-40,41,1),alpha=0.5,histtype='bar',ec='black',align='mid',log=True)

                ax.set_xlim(-40,40)
                ax.set_ylim(1e0,1e5)

                ax.set_xlabel('Velocity Difference from Mean Difference (m s$\mathregular{^{-1}}$)',fontsize=10)

                fig.savefig(name+'before.png',dpi=300)

            #Initialize arrays for staggered PRT correction.
            #Also set "bound" which is the width of the search for errors.
            #fnyq = VNyqHigh + VNyqLow
            if staggeredPRT:
                possibleSolutions = np.empty((16,velNew.shape[0],velNew.shape[1]))
                differences = np.empty((16,velNew.shape[0],velNew.shape[1]))
                bound = fnyq

            #Initialize arrays for dual PRF correction.
            if dualPRF:
                possibleSolutions = np.empty((7,velNew.shape[0],velNew.shape[1]))
                differences = np.empty((7,velNew.shape[0],velNew.shape[1]))

            #First solution is the original velocity field.
            possibleSolutions[0,:] = velNew.copy()
            mask = np.zeros(velNew.shape)
            
            #Initialize count = 1 because of prior entry of original velocity field.
            count = 1

            #Staggered PRT Correction for all solutions.
            if staggeredPRT:
                #Loop over n1, n2 = 0,1,2,3.
                for n1 in [0,1,2,3]:
                    for n2 in [0,1,2,3]:
                        #Error is sum of n1(VNLow) + n2(VNHigh)
                        if ((n1 == 0) & (n2 == 0)):
                            continue
                        nyq = n1*nyqL + n2*nyqH                  

                        #Copy original/partially corrected field.
                        velPossible = velNew.copy()

                        #Set the lower bound for search. Cannot cross 0.
                        if nyq-bound < 0: limit = 0
                        else: limit=nyq-bound
                
                        #Search for positive errors where azimuthal AND radial differences are greater than
                        #3 standard deviations and meet error criteria.
                        positiveIndices = np.where(((np.isnan(diffAz) != True) & (np.isnan(diffRad) != True)) & \
                            (diffMean > limit) & (diffMean < nyq + bound))

                        #Correct positive errors.
                        velPossible[positiveIndices] = velPossible[positiveIndices] - nyq
                        mask[positiveIndices] = 1
                        
                        #As above, but for negative errors.
                        negativeIndices = np.where(((np.isnan(diffAz) != True) & (np.isnan(diffRad) != True)) & \
                            (diffMean < -1*(limit)) & (diffMean > -1*(nyq + bound)))

                        #Correct negative errors.
                        velPossible[negativeIndices] = velPossible[negativeIndices] + nyq
                        mask[negativeIndices] = 1

                        #Save solution for n1, n2.
                        possibleSolutions[count,:] = velPossible
                        
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

                        #Set the lower bound for search. Cannot cross 0.
                        if nyq-bound < 0: limit = 0
                        else: limit=nyq-bound

                        #As in staggered PRT correction method, check that gates meet error criteria for
                        #positive errors.
                        positiveIndices = np.where(((np.isnan(diffAz) != True) & (np.isnan(diffRad) != True)) & \
                            (diffMean > limit) & (diffMean < nyq + bound))

                        #Correct positive errors.
                        velPossible[positiveIndices] = velPossible[positiveIndices] - nyq
                        mask[positiveIndices] = 1
                        
                        #Check for error criteria for negative errors.
                        negativeIndices = np.where(((np.isnan(diffAz) != True) & (np.isnan(diffRad) != True)) & \
                            (diffMean < -1*(limit)) & (diffMean > -1*(nyq + bound)))

                        #Correct negative errors.
                        velPossible[negativeIndices] = velPossible[negativeIndices] + nyq
                        mask[negativeIndices] = 1

                        #Save possible solution for n1 and nyquist.
                        possibleSolutions[count,:] = velPossible

                        #Compute difference of solution relative to smoothed mean field.
                        differences[count,:] = velPossible-velSmoothMean

                        #On to the next one.
                        count+=1

            velSmoothRecompute = np.nanmean(np.array([smoothVel(np.ma.masked_where(mask==1,velNew),radar.range['data'],radFilter)[0],\
                            smoothVel(np.ma.masked_where(mask==1,velNew).T,radar.azimuth['data'][sweep_slice],azFilter)[0].T]),axis=0)
            differences = np.array([velPoss - velSmoothRecompute for velPoss in possibleSolutions])

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
    
    if plotStats == True:
        fig = plt.figure(1)
        fig.clf()
        gs1 = gridspec.GridSpec(1,1)
        gs1.update(wspace=0.2,hspace=0.2)
        fig.set_size_inches(5,5)

        ax = fig.add_subplot(gs1[0])

        a=ax.hist(diffMean.flatten(),range(-40,41,1),alpha=0.5,histtype='bar',ec='black',align='mid',log=True)

        ax.set_xlim(-40,40)
        ax.set_ylim(1e0,1e5)

        ax.set_xlabel('Velocity Difference from Mean Difference (m s$\mathregular{^{-1}}$)',fontsize=10)

        fig.savefig(name+'after.png',dpi=300)
    
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