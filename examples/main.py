import pyart
from processorCorrect import velocityCorrection
import glob
import numpy as np
import matplotlib.pyplot as plt
import ctables
try: import raddison
except: 
    print("Importing Raddison failed. Please contact Addison Alford (addisonalford@ou.edu)")
    print("if you would like to use the Python raddison package. It is still under development.")

import pyart

def radarObjCorrect(radar):
    '''
    +++ Description +++

    Reorders the azimuth dimension to correct incorrectly recorded azimuths
    such that all sweeps are in ascending order as a function of azimuth.

    Example:

    import pyart
    radar = pyart.io.read('cfrad.blah.blah.nc')
    radar = radarObjCorrect(radar)

    Happy plotting/dealiasing!

    +++ Input +++

    radar:  The radar object from py-art.

    +++ Returns +++

    radar
    
    '''
    
    for sweep in radar.iter_slice():

        indices = radar.time['data'][sweep].argsort()
        radar.time['data'][sweep] = radar.time['data'][sweep][indices]

        radar.azimuth['data'][sweep] = radar.azimuth['data'][sweep][indices]
        radar.elevation['data'][sweep] = radar.elevation['data'][sweep][indices]

        for key in radar.fields.keys():
            radar.fields[key]['data'][sweep] = radar.fields[key]['data'][sweep][indices]

    radar.init_gate_x_y_z()
    radar.init_gate_longitude_latitude()
    radar.init_gate_altitude()
    
    return radar

radarCPOL = {'name':'CPOL','ref':'DBZ','vel':'VEL'}
radarDOW = {'name':'DOW6','ref':'DZ','vel':'VEL'}
radarP3 = {'name':'N42R','ref':'DBZ','vel':'VD'}

radars = [radarCPOL,radarDOW,radarP3]

for f in sorted(glob.glob('./data/cfrad.*')):
    if '.png' in f: continue

    radar = pyart.io.read(f)

    radarName = f.split('/')[-1].split('_')[5]

    for r in radars:
        if r['name'] == radarName:
            radarDict = r
    print(f)
    if 'VTO' not in radar.fields.keys():
        try:
            if 'DOW' in radarName:
                if np.nanmean(radar.elevation['data'][:]) > 2.:
                    radar.fields['VEL']['data'] = np.ma.masked_where(radar.fields['DBZHC']['data']<-10,\
                        radar.fields['VEL']['data'])
                radar.add_field_like('VEL','VTO',radar.fields['VEL']['data'].copy())
            elif 'CPOL' in radarName:
                radar.fields['VRo']['data'] = np.ma.masked_where(radar.fields[radarDict['ref']]['data']<20,\
                    radar.fields['VRo']['data'])
                radar.add_field_like('VRo','VTO',radar.fields['VRo']['data'].copy())
            elif 'N42R' in radarName:
                radar.add_field_like('VD','VTO',radar.fields['VD']['data'].copy())
            else:
                radar.add_field_like('VRo','VTO',radar.fields['VRo']['data'].copy())
            print(f,"Done")
        except:
            if 'CPOL' in radarName:
                radar.fields['VEL']['data'] = np.ma.masked_where(radar.fields[radarDict['ref']]['data']<20,\
                    radar.fields['VEL']['data'])
            radar.add_field_like('VEL','VTO',radar.fields['VEL']['data'].copy())
    
    if 'VTC' not in radar.fields.keys():
        radar.add_field_like('VTO','VTC',radar.fields['VTO']['data'].copy())

    vHigh,vLow,fNyq = velocityCorrection.getNyq(radar,radarName)

    print(vHigh,vLow,fNyq)

    if 'N42R' in radarName:
        radar = velocityCorrection.errorCorrect(radar,velField='VTC',nyqL=vHigh,nyqH=vLow,
            method='dual',name=f[:-3])

    elif 'CPOL' in radarName:
        radar.fields['VTC'] = pyart.correct.dealias_region_based(radar,vel_field='VTC',
            nyquist_vel=48.)

        radar = velocityCorrection.errorCorrect(radar,velField='VTC',fnyq=fNyq,nyqL=vHigh,
            nyqH=vLow,method='staggered',name=f[:-3])

    elif 'DOW' in radarName:
        radar = velocityCorrection.errorCorrect(radar,velField='VTC',fnyq=fNyq,nyqL=vHigh,
            nyqH=vLow,method='staggered',name=f[:-3])

    else:
        radar.fields['VTC'] = pyart.correct.dealias_region_based(radar,vel_field='VTC')
        radar = velocityCorrection.errorCorrect(radar,velField='VTC',fnyq=fNyq,nyqL=vHigh,
            nyqH=vLow,method='staggered',name=f[:-3])

    radar_file_new = f[:-3]+'.png'

    azgrid,rgrid = np.meshgrid(radar.azimuth['data'],radar.range['data']/1000.,indexing='ij')

    x,y = rgrid*np.sin(np.radians(azgrid)),rgrid*np.cos(np.radians(azgrid))
    '''fig = plt.figure(1)
    fig.set_size_inches(10,5)
    plt.clf()
    for n,vel,title in \
     zip(range(121,123,1),[radar.fields['VTO']['data'][:],radar.fields['VTC']['data'][:]],['Original','Corrected']):
    
        ax = fig.add_subplot(n)

        a = ax.pcolormesh(x.T,y.T,vel,vmin=-50,vmax=50,cmap=ctables.Carbone42)

        ax.set_title(title)

        #ax.set_xlim(-20,0)
        #ax.set_ylim(0,20)

        plt.colorbar(a)

        plt.savefig(radar_file_new,dpi=600)'''
    
    refRange = np.arange(10,60,5)
    #velRange = np.arange(-30,31,5)
    velRange = np.arange(-40,41,5)
    el = np.nanmean(radar.get_elevation(0))
    date,time = f.split('.')[2].split('_')
    xLabels,yLabels = range(-200,201,10),range(-200,201,10)

    if radarDict['ref'] not in radar.fields.keys():
        if 'DZ' in radar.fields.keys():
            radarDict['ref'] = 'DZ'
        elif 'DBZ' in radar.fields.keys():
            radarDict['ref'] = 'DBZ'
        elif 'DBZHC' in radar.fields.keys():
            radarDict['ref'] = 'DBZHC'
        else:
            raise KeyError("Having trouble finding reflectivity variable. Tried %s, DZ, DBZ, and DBZHC."%radarDict['ref'])

    if ('DOW' in radarName) and (el < 2):
        #xMin,yMin = -80,-80
        #xMax,yMax = 80,80
        xMin,yMin = -40,25
        xMax,yMax = 10,75
    elif ('DOW' in radarName) and (el > 2):
        #xMin,yMin = -50,-50
        #xMax,yMax = 50,50
        #xMin,yMin = 10,25
        #xMax,yMax = 35,50
        xMin,yMin = 5,-40
        xMax,yMax = 50,5

    elif ('CPOL' in radarName) and ('2009' in date):
        xMin,yMin = -10,10
        xMax,yMax = 20,40

        #xMin,yMin = -10,5
        #xMax,yMax = 5,20
    elif ('CPOL' in radarName) and ('2012' in date):
        xMin,yMin = 0,-40
        xMax,yMax = 80,40
    elif 'N42' in radarName:
        xMin,yMin = -25,0
        xMax,yMax = 0,25
    else:
        xMin,xMax,yMin,yMax = None,None,None,None

    raddison.mapping.meshPlots.ppiPColormesh(spaceX=x,
        spaceY=y,
        toContourFill=[radar.fields['VTO']['data'],radar.fields['VTC']['data'][:]],
        toQuiver=[None,None],
        toContour=None,
        ranges=[velRange,velRange],
        contourRanges=None,
        titles=['Uncorrected V$\mathregular{_R}$ (m s$\mathregular{^{-1}}$)',
            'Corrected V$\mathregular{_R}$ (m s$\mathregular{^{-1}}$)'],
        supTitle = ' %s %s UTC | %.1f deg PPI'%(date,time,el) ,
        xTicks={'ticks':xLabels,'labels':xLabels},
        yTicks={'ticks':xLabels,'labels':yLabels},
        cbPlot=[True,False,False],
        units=['m s$\mathregular{^{-1}}$','m s$\mathregular{^{-1}}$'],
        cbPos = [1,1,2],
        cbType = [ctables.Carbone42,ctables.Carbone42],
        cbLabels = [velRange,velRange],
        trackPoints=None,
        radarPoints=None,
        instrumentPoints=None,
        path='./data/',
        savePath='./data/'+'ppi_%s_%s_%s_dbz_vel_%.1fdeg_uncorrected_zoom1.png' %(radarName,date,time,el),
        qt = None,
        xlim=(xMin,xMax),
        ylim=(yMin,yMax))

    '''raddison.mapping.meshPlots.ppiPColormesh(spaceX=x,
        spaceY=y,
        toContourFill=[radar.fields[radarDict['ref']]['data'],radar.fields['VTO']['data'][:]],
        toQuiver=[None,None],
        toContour=None,
        ranges=[refRange,velRange],
        contourRanges=None,
        titles=['Z$\mathregular{_H}$ (dBZ)',
            'Uncorrected V$\mathregular{_R}$ (m s$\mathregular{^{-1}}$)'],
        supTitle = ' %s %s UTC | %.1f deg PPI'%(date,time,el) ,
        xTicks={'ticks':xLabels,'labels':xLabels},
        yTicks={'ticks':xLabels,'labels':yLabels},
        cbPlot=[True,True,False],
        units=['dBZ','m s$\mathregular{^{-1}}$'],
        cbPos = [1,2,2],
        cbType = [ctables.Carbone42,ctables.BuDRd18],
        cbLabels = [refRange,velRange],
        trackPoints=None,
        radarPoints=None,
        instrumentPoints=None,
        path='./data/',
        savePath='./data/'+'ppi_%s_%s_%s_dbz_vel_%.1fdeg_uncorrected_zoom1.png' %(radarName,date,time,el),
        qt = None,
        xlim=(xMin,xMax),
        ylim=(yMin,yMax))'''

    '''raddison.mapping.meshPlots.ppiPColormesh(spaceX=x,
        spaceY=y,
        toContourFill=[radar.fields[radarDict['ref']]['data'],radar.fields['VTC']['data'][:]],
        toQuiver=[None,None],
        toContour=None,
        ranges=[refRange,velRange],
        contourRanges=[range(10,100,10),range(10,100,10)],
        titles=['Z$\mathregular{_H}$ (dBZ)',
            'Corrected V$\mathregular{_R}$ (m s$\mathregular{^{-1}}$)'],
        supTitle = ' %s %s UTC | %.1f deg PPI'%(date,time,el) ,
        xTicks={'ticks':xLabels,'labels':xLabels},
        yTicks={'ticks':xLabels,'labels':yLabels},
        cbPlot=[True,True],
        units=['dBZ','m s$\mathregular{^{-1}}$'],
        cbPos = [1,2],
        cbType = [ctables.Carbone42,ctables.BuDRd18],
        cbLabels = [refRange,velRange],
        trackPoints=None,
        radarPoints=None,
        instrumentPoints=None,
        path='./data/',
        savePath='./data/'+'ppi_%s_%s_%s_dbz_vel_%.1fdeg_corrected.png' %(radarName,date,time,el),
        qt = None,
        xlim=(xMin,xMax),
        ylim=(yMin,yMax))'''