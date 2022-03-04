from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def smoothVel(vel,spatial,filter):

    #Smooth interpolated field.
    velSmooth = savgol_filter(vel, filter, 3, mode='interp',axis=1)

    #Change -32768 to NaNs.
    velSmooth[velSmooth < -1000] = np.nan

    #Create difference field
    diff = vel.filled(fill_value=np.nan) - velSmooth

    return velSmooth,diff

def prtCorrectNew(velField,radius,azimuth,fnyq = 0,nyqL=0,nyqH=0):

    '''
    +++ Description +++

    This function corrects errors related to staggered-PRT processing, is copied
    from ../velocityCorrection.py, and is modified to take in velField for testing.

    Please see the errorCorrection function in ../velocityCorrection.py for comments and 
    descriptions. 

    '''

    pointsCaught = 0
    count = 0
    for radFilter,azFilter in zip([11,21,5,51,71,5],[5,9,5,21,71,5]):
        pointsCaughtPrev = pointsCaught            
        #Radial
        
        vel = velField.copy()
        r = radius
        
        velInterp = vel
        
        velSmooth = savgol_filter(velInterp, radFilter, 3, mode='interp',axis=1)
        
        velSmoothRad = velSmooth

        diffRad = vel - velSmooth
        
        #### Azimuthal ####
        vel = velField.copy().T
        r = azimuth
        
        velInterp = vel
        
        velSmooth = savgol_filter(velInterp, azFilter, 3, mode='interp',axis=1)

        velSmoothAz = velSmooth.T

        velSmoothMean = np.nanmean(np.array([velSmoothAz,velSmoothRad]),axis=0)
        
        diffAz = vel.T - velSmooth.T

        diffMean = vel.T - velSmoothMean

        diff = np.nanmax(np.array([diffRad,diffAz]),axis=0)
        
        velNew = vel.copy().T

        mu,stdAz = norm.fit(np.ma.masked_invalid(diffAz).compressed())
            
        #Compute the standard deviation of the radial difference field.
        mu,stdRad = norm.fit(np.ma.masked_invalid(diffRad).compressed())

        if (max([nyqL,nyqH])) < 3*stdRad and (max([nyqL,nyqH])) < 3*stdAz:
            radMask,azMask = min([nyqL,nyqH]),min([nyqL,nyqH])
        else:
            radMask,azMask = 3*stdRad,3*stdAz

        #Fill NaNs where azimuthal difference is < 3*(standard deviation).
        diffAz = np.ma.masked_where(np.abs(diffAz) < azMask,diffAz).filled(fill_value = np.nan)
        
        #Fill NaNs where radial difference is < 3*(standard deviation).
        diffRad = np.ma.masked_where(np.abs(diffRad) < radMask,diffRad).filled(fill_value = np.nan) 
        
        possibleSolutions = np.empty((16,velNew.shape[0],velNew.shape[1]))
        differences = np.empty((16,velNew.shape[0],velNew.shape[1]))
        possibleSolutions[0,:] = velNew.copy()
        differences[0,:] = velNew.copy() - velSmoothMean
        mask = np.zeros(velNew.shape)
        
        count = 1
        bound = fnyq
        for n1 in [0,1,2,3]:
            for n2 in [0,1,2,3]:
                if ((n1 == 0) & (n2 == 0)):
                    continue
                nyq = n1*nyqL + n2*nyqH                  
                #### Both
                velPossible = velNew.copy()
                
                if nyq-bound < 0: limit = 0
                else: limit=nyq-bound
                positiveIndices = np.where(((np.isnan(diffAz) != True) & (np.isnan(diffRad) != True)) & \
                    (diffMean > limit) & (diffMean < nyq + bound))
                mask[positiveIndices] = 1
                velPossible[positiveIndices] = velPossible[positiveIndices] - nyq
                
                negativeIndices = np.where(((np.isnan(diffAz) != True) & (np.isnan(diffRad) != True)) & \
                    (diffMean < -1*limit) & (diffMean > -1*(nyq + bound)))
                velPossible[negativeIndices] = velPossible[negativeIndices] + nyq
                mask[negativeIndices] = 1

                possibleSolutions[count,:] = velPossible

                count+=1

        velSmoothRecompute = np.nanmean(np.array([smoothVel(np.ma.masked_where(mask==1,velNew),radius,radFilter)[0],\
                smoothVel(np.ma.masked_where(mask==1,velNew).T,azimuth,azFilter)[0].T]),axis=0)
        differences = np.array([velPoss - velSmoothRecompute for velPoss in possibleSolutions])

        differences = np.abs(np.ma.masked_invalid(differences).filled(fill_value=0.))

        azimuths,ranges = np.meshgrid(range(velSmoothMean.shape[0]),range(velSmoothMean.shape[1]),indexing='ij')
        
        indices = tuple([np.nanargmin(differences,axis=0).flatten(),azimuths.flatten(),ranges.flatten()])

        pointsCaught = pointsCaught + np.where(np.nanargmin(differences,axis=0).flatten()!=0)[0].shape[0]

        finalSolution = possibleSolutions[indices].reshape(velSmoothMean.shape)

        velField = finalSolution         
        
    print("TOTAL POINTS CAUGHT ",pointsCaught)
    
    return velField