from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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

        diffAz = np.ma.masked_where(np.abs(diffAz) < 3*stdAz,diffAz).filled(fill_value = np.nan)

        mu,stdRad = norm.fit(np.ma.masked_invalid(diffRad).compressed())

        print(stdAz,stdRad)

        diffRad = np.ma.masked_where(np.abs(diffRad) < 3*stdRad,diffRad).filled(fill_value = np.nan)
        
        possibleSolutions = np.empty((10,velNew.shape[0],velNew.shape[1]))
        differences = np.empty((10,velNew.shape[0],velNew.shape[1]))
        possibleSolutions[0,:] = velNew.copy()
        differences[0,:] = velNew.copy() - velSmoothMean
        
        count = 1
        bound = fnyq
        for n1 in [1,2,3]:
            for n2 in [1,2,3]:
                nyq = n1*nyqL + n2*nyqH                  
                #### Both
                velPossible = velNew.copy()
                positiveIndices = np.where((diffAz != np.nan) & (diffRad != np.nan) & \
                    (diffMean > nyq - bound) & (diffMean < nyq + bound))
                velPossible[positiveIndices] = velPossible[positiveIndices] - nyq
                
                negativeIndices = np.where((diffAz != np.nan) & (diffRad != np.nan) & \
                    (diffMean < -1*(nyq - bound)) & (diffMean > -1*(nyq + bound)))
                velPossible[negativeIndices] = velPossible[negativeIndices] + nyq

                possibleSolutions[count,:] = velPossible
                differences[count,:] = velPossible-velSmoothMean

                count+=1

        differences = np.abs(np.ma.masked_invalid(differences).filled(fill_value=0.))

        azimuths,ranges = np.meshgrid(range(velSmoothMean.shape[0]),range(velSmoothMean.shape[1]),indexing='ij')
        
        indices = tuple([np.nanargmin(differences,axis=0).flatten(),azimuths.flatten(),ranges.flatten()])

        pointsCaught = pointsCaught + np.where(np.nanargmin(differences,axis=0).flatten()!=0)[0].shape[0]

        finalSolution = possibleSolutions[indices].reshape(velSmoothMean.shape)

        velField = finalSolution         
        
    print("TOTAL POINTS CAUGHT ",pointsCaught)
    
    return velField