import pyart
from processorCorrect import velocityCorrection
import glob
import numpy as np
import matplotlib.pyplot as plt
import ctables

radarCPOL = {'name':'CPOL','ref':'DBZ','vel':'VEL'}
radarDOW = {'name':'DOW6','ref':'DZ','vel':'VEL'}
radarP3 = {'name':'N42R','ref':'DBZ','vel':'VD'}

radars = [radarCPOL,radarDOW,radarP3]

for f in sorted(glob.glob('./data/cfrad.2015*')):
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
            method='dual')

    elif 'CPOL' in radarName:
        radar.fields['VTC'] = pyart.correct.dealias_region_based(radar,vel_field='VTC',
            nyquist_vel=48.)

        radar = velocityCorrection.errorCorrect(radar,velField='VTC',fnyq=fNyq,nyqL=vHigh,
            nyqH=vLow,method='staggered')

    elif 'DOW' in radarName:
        radar = velocityCorrection.errorCorrect(radar,velField='VTC',fnyq=fNyq,nyqL=vHigh,
            nyqH=vLow,method='staggered')

    else:
        radar.fields['VTC'] = pyart.correct.dealias_region_based(radar,vel_field='VTC')
        radar = velocityCorrection.errorCorrect(radar,velField='VTC',fnyq=fNyq,nyqL=vHigh,
            nyqH=vLow,method='staggered')

    radar_file_new = f[:-3]+'.png'

    rgrid,azgrid = np.meshgrid(radar.range['data']/1000.,radar.azimuth['data'],indexing='ij')

    x,y = rgrid*np.sin(np.radians(azgrid)),rgrid*np.cos(np.radians(azgrid))

    fig = plt.figure(1)
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

        plt.savefig(radar_file_new,dpi=600)