#!/usr/bin/python3.8
import sys,os
import numpy as np
from netCDF4 import Dataset,num2date
from PV import potential_vorticity
from cdo import Cdo
from nco import Nco
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import matplotlib as mpl
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import math
import datetime
import xarray as xr
import re
import time
def main():


    cdo = Cdo()

    cdo.remapbil("myGridDef",input="surface_temp_2023_10_8_00Z.nc",output="surface_temp_10_8_2023.nc")
    cdo.remapbil("myGridDef",input="surface_uwnd_2023_10_8_00Z.nc",output="surface_uwnd_10_8_2023.nc")
    cdo.remapbil("myGridDef",input="surface_vwnd_2023_10_8_00Z.nc",output="surface_vwnd_10_8_2023.nc")


    startTime = time.time()

    
    tmp_file_list = [ file for file in os.listdir('.') if file.startswith("air") ]
    uwnd_file_list = [ file for file in os.listdir('.') if file.startswith("uwnd") ]
    vwnd_file_list = [ file for file in os.listdir('.') if file.startswith("vwnd") ]

    fileTmpDictionary = {}
    for file in tmp_file_list:
        pressureLevel = int(file.split("_")[1])
        fileTmpDictionary[pressureLevel] = file

    cdo.merge(input=" ".join(([fileTmpDictionary[key] for key in sorted(fileTmpDictionary,reverse=True)])), output='tmpFile.nc')
    fileUwndDictionary = {}
    for file in uwnd_file_list:
        pressureLevel = int(file.split("_")[1])
        fileUwndDictionary[pressureLevel] = file

    cdo.merge(input=" ".join(([fileUwndDictionary[key] for key in sorted(fileUwndDictionary,reverse=True)])), output='uwndFile.nc')
    fileVwndDictionary = {}
    for file in vwnd_file_list:
        pressureLevel = int(file.split("_")[1])
        fileVwndDictionary[pressureLevel] = file

    cdo.merge(input=" ".join(([fileVwndDictionary[key] for key in sorted(fileVwndDictionary,reverse=True)])), output='vwndFile.nc')


    cdo.merge(input=" ".join(('tmpFile.nc','uwndFile.nc','vwndFile.nc','surface_temp_10_8_2023.nc','pres_sfc_2023_10_8_00Z.nc','surface_uwnd_10_8_2023.nc','surface_vwnd_10_8_2023.nc')),output='pvFile.nc')

    if __name__ == "__main__":
        #client = Client()
        ds_pv = openFile()
        executeCalc(ds_pv)
        
def openFile():
            
    ds_pv = xr.open_mfdataset("pvFile.nc",chunks={'time':1})
    print(ds_pv)
    return ds_pv

def executeCalc(ds_pv):
    levels = ds_pv.coords['level'].values
    plevs = np.array(levels)
    plevs = plevs*100
    print(plevs)
    
    ipvArgs = []
    uArgs = []
    
    vArgs= []
    pv = potential_vorticity()    
    tmp = ds_pv.air.values
    tsfc = ds_pv.air_2.values
    psfc = ds_pv.pres.values
    uwnd= ds_pv.uwnd_2.values
    vwnd = ds_pv.vwnd_2.values
    u = ds_pv.uwnd.values
    v = ds_pv.vwnd.values
    
    lats = ds_pv.coords['lat'].values
    lons = ds_pv.coords['lon'].values
    dates  = ds_pv.time.values

    dates = dates.astype(str)
    
    dates=np.array(list(map(lambda v:re.sub('T',' ',v),dates)))
    dates=np.array(list(map(lambda v:re.sub('\.[0]+',' ',v),dates)))
    print(dates)
    
    dateMean = ds_pv.time.mean()
    dateMean = (dateMean.values)
    dateMean = np.datetime_as_string(dateMean)
    dateMean = re.sub('T0',' ',dateMean)
    dateMean = re.sub('\.[0]+',' ',dateMean)
    
    missingData = -999.99
    for i in range(0,tmp.shape[0]):

        tmpInstant = tmp[i,:,:,:]

        tsfcInstant = tsfc[i,:,:]
        psfcInstant = psfc[i,:,:]
        ret = pv.p2thta(lats,lons,plevs,tsfcInstant,psfcInstant,tmpInstant)
        kthta,pthta,thta = ret
        isent = np.where(thta == 360)[0]
        isent = isent[0]
        uwndInstant = uwnd[i,:,:]
        vwndInstant = vwnd[i,:,:]
        uInstant = u[i,:,:,:]
        vInstant = v[i,:,:,:]
        uthta = pv.s2thta(lats,lons,plevs,kthta,uwndInstant,psfcInstant,uInstant,thta,pthta)

        vthta = pv.s2thta(lats,lons,plevs,kthta,vwndInstant,psfcInstant,vInstant,thta,pthta)
        ipvInstant = pv.sipv2(lats,lons,kthta,thta,pthta,uthta,vthta,missingData)
        
        ipvPlot = ipvInstant[isent,:,:]
        print(ipvPlot.shape)
        uipvPlot = uthta[isent,:,:]
        vipvPlot = vthta[isent,:,:]
        ipvMeridional = pv.ddy(ipvPlot,lats,lons)
        plotIPV(lats,lons,ipvPlot,uipvPlot,vipvPlot,ipvMeridional,dates[i],temp=360)
        ipv3d = np.expand_dims(ipvPlot,0)
        print(ipv3d.shape)
        uthta3d = np.expand_dims(uipvPlot,0)
        vthta3d = np.expand_dims(vipvPlot,0)
        uArgs.append(uthta3d)
        vArgs.append(vthta3d)
        ipvArgs.append(ipv3d)
    ipv = np.concatenate(ipvArgs,axis=0)
    uipv = np.concatenate(uArgs,axis=0)
    vipv = np.concatenate(vArgs,axis=0)
    print(uipv.shape,vipv.shape,ipv.shape)
    ipvMean = np.mean(ipv,axis=0)
    print(ipvMean.shape)
    uipvMean = np.mean(uipv,axis=0)
    vipvMean = np.mean(vipv,axis=0)
    print(uipvMean.shape,vipvMean.shape)

    for i in range (0,len(lons)):
        if (lons[i] < 0.0):
            lons[i] += 360.0
    #ipvPlot = np.empty((nj,ni))
    #ipvPlot = ipvMean[11,:,:]
    #uipvPlot = uipvMean[11,:,:]
    #vipvPlot = vipvMean[11,:,:]
    #print(ipvPlot.shape)
            #s = Spharmt(len(lons),len(lats), gridtype='regular',
            #                         rsphere=6.3712e6, legfunc='stored')

    #ipv3d = np.expand_dims(ipvPlot,axis=2)
    #print(ipv3d.shape)
    #ipvZonal,ipvMeridional = s.getgrad(ipv3d)
    #print(ipvZonal.shape,ipvMeridional.shape)
    #print(ipvMeridional.shape)    
    #sys.exit(
    ipvMeridional = pv.ddy(ipvMean,lats,lons)



    print(max(ipvMean.flatten()))
    print(min(ipvMean.flatten()))


    plotIPV(lats,lons,ipvMean,uipvMean,vipvMean,ipvMeridional,dateMean,temp=360)
    stopTime = time.time()
    print(stopTime-startTime)
    
def plotIPV(lats,lons,ipvPlot,uipvPlot,vipvPlot,ipvMeridional,date,temp):

    uipvPlot = uipvPlot[::3,::3]
    vipvPlot = vipvPlot[::3,::3]
    np.set_printoptions(threshold=sys.maxsize)
    #print(uipvPlot)
    #print(vipvPlot)
    ax1 = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    clevs = np.arange(-11,11,1.0)
    shear_fill = ax1.contourf(lons,lats,ipvPlot,clevs,
                              transform=ccrs.PlateCarree(), cmap=plt.get_cmap('hsv'),
                              extend='both')
    line_c = ax1.contour(lons, lats, ipvPlot, levels=[1.5,2.0,6.0,8.0],
                         colors=['red','black','yellow','white'],
                         transform=ccrs.PlateCarree())

    line_ipvgrad = ax1.contour(lons,lats,ipvMeridional,colors=['white'],transform=ccrs.PlateCarree())
    lons = lons[::3]
    lats = lats[::3]
    #print(lons.shape,lats.shape,uipvPlot.shape,vipvPlot.shape)
    #ax1.quiver(lons,lats,uipvPlot,vipvPlot,transform=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.gridlines()
    ax1.set_xticks([0, 60, 120, 180, 240, 300, 359.99], crs=ccrs.PlateCarree())
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True,
                                       number_format='.0f')
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    cbar = plt.colorbar(shear_fill, orientation='horizontal')
    #date = date.strftime('%Y-%m-%d-%H')
    print(date)
    isent = str(temp)
    plt.title('PV '+ isent+'K surface '+ date, fontsize=16)
    plt.savefig(date+'_spec'+isent+'K.png')
    plt.show()

            
main()
