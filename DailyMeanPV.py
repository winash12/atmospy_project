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
from windspharm.standard import VectorWind
from spharm import Spharmt
import math
import datetime

def main():


    cdo = Cdo()
    nco = Nco()
    

    cdo.remapbil("myGridDef",input="surface_temp_2021_23_2_00Z.nc",output="surface_temp_21_2_2021.nc")
    cdo.remapbil("myGridDef",input="surface_uwnd_2021_23_2_00Z.nc",output="surface_uwnd_21_2_2021.nc")
    cdo.remapbil("myGridDef",input="surface_vwnd_2021_23_2_00Z.nc",output="surface_vwnd_21_2_2021.nc")


    
    pv = potential_vorticity()
    
    pres = ["100000","92500","85000","70000","60000","50000","40000","30000","25000","20000","15000","10000","7000","5000","3000","2000","1000"]
    tmp_file_list = [ file for file in os.listdir('.') if file.startswith("air") ]
    uwnd_file_list = [ file for file in os.listdir('.') if file.startswith("uwnd") ]
    vwnd_file_list = [ file for file in os.listdir('.') if file.startswith("vwnd") ]
    #print(tmp_file_list)

    #print(uwnd_file_list)
    #print(vwnd_file_list)

    #Start
    tmpFilePressureLevelDictionary={}
    for file in tmp_file_list:
        pressureLevel = int(file.split("_")[1])
        tmpFilePressureLevelDictionary[pressureLevel] = file

    args = []
    args1 = []
    missing  = 0
    time = 0
    for key in  sorted(tmpFilePressureLevelDictionary,reverse=True):
        file = tmpFilePressureLevelDictionary[key]
        nc_tempFile = Dataset(file,'r')
        lats = nc_tempFile.variables['lat'][:]  # extract/copy the data
        lons = nc_tempFile.variables['lon'][:]
        time = nc_tempFile.variables['time'][:]
        dates = num2date(time, nc_tempFile.variables['time'].units)
        lats = lats[:].squeeze()
        lons = lons[:].squeeze()
        nj = len(lats)
        ni = len(lons)

        for name,variable in nc_tempFile.variables.items():
            for attrname in variable.ncattrs():
                try:
                    if attrname ==  "missing_value":
                        missingData = variable.getncattr(attrname)
                        break
                except:
                    print("Missing Data Not Found")
        temp = nc_tempFile.variables['air'][:]
        #print(temp.shape)
        #temp_theta = nc_tempFile.variables['air'][0,:]*units(nc_tempFile.variables['air'].units)
        # Reanalysis data is oriented N-S. So reverse the array
        #temp = temp[::-1,:,:]
        args.append(temp)
        #args1.append(temp_theta)
    tmp = np.concatenate(args,axis=1)
    print(tmp.shape)
    #tmp_theta = np.concatenate(args1,axis=0)

    #Start Uwind
    uwndFilePressureLevelDictionary={}
    for file in uwnd_file_list:
        pressureLevel = int(file.split("_")[1])
        uwndFilePressureLevelDictionary[pressureLevel] = file
        args = []
        args1 = []
    for key in  sorted(uwndFilePressureLevelDictionary,reverse=True):
        file = uwndFilePressureLevelDictionary[key]
        nc_uwndFile = Dataset(file,'r')
        uwnd = nc_uwndFile.variables['uwnd'][:]
        #uwnd_theta = nc_uwndFile.variables['uwnd'][0,:]*units(nc_uwndFile.variables['uwnd'].units)
        args.append(uwnd)
        #args1.append(uwnd_theta)
    u = np.concatenate(args,axis=1)
    #u_theta = np.concatenate(args1,axis=0)
    #Done
    print(u.shape)

        
    #Start VWind
    vwndFilePressureLevelDictionary={}
    for file in vwnd_file_list:
        #        print(file)
        pressureLevel = int(file.split("_")[1])
        vwndFilePressureLevelDictionary[pressureLevel] = file
        args = []
        args1 = []
    for key in  sorted(vwndFilePressureLevelDictionary,reverse=True):
        #        print(key)
        file = vwndFilePressureLevelDictionary[key]
        nc_vwndFile = Dataset(file,'r')
        vwnd = nc_vwndFile.variables['vwnd'][:]
        #vwnd_theta = nc_vwndFile.variables['vwnd'][0,:]*units(nc_vwndFile.variables['vwnd'].units)
        args.append(vwnd)
        #args1.append(vwnd_theta)
    v = np.concatenate(args,axis=1)
    #v_theta = np.concatenate(args1,axis=0)
    print(v.shape)

    #Done

    tmp_file = "surface_temp_2021_23_2_00Z.nc"
    nc_surfTFile = Dataset(tmp_file,'r')
    tsfc = nc_surfTFile.variables['air'][:]
    print(tsfc.shape)
    
    pres_file = "pres_sfc_2021_23_2_00Z.nc"
    nc_presFile = Dataset(pres_file,'r')
    psfc = nc_presFile.variables['pres'][:]
    print(psfc.shape)

    uwnd_10m_file = "surface_uwnd_2021_23_2_00Z.nc"
    nc_uwndFile = Dataset(uwnd_10m_file,'r')
    uwnd = nc_uwndFile.variables['uwnd'][:]
    print(uwnd.shape)
    
    vwnd_10m_file = "surface_vwnd_2021_23_2_00Z.nc"
    nc_vwndFile = Dataset(vwnd_10m_file,'r')
    vwnd = nc_vwndFile.variables['vwnd'][:]
    print(vwnd.shape)

    plevs = np.array(pres)
    plevs = plevs.astype(np.float)
    ipvArgs = []
    uArgs = []
    vArgs= []
    for i in range(0,tmp.shape[0]):

        print(tmp[i,:,:,:].shape)
        tmpInstant = tmp[i,:,:,:]
        tsfcInstant = tsfc[i,:,:]
        psfcInstant = psfc[i,:,:]
        ret = pv.p2thta(lats,lons,plevs,tsfcInstant,psfcInstant,tmpInstant)
        kthta,pthta,thta = ret
        #print(thta.shape)
        print(thta)
        isent = np.where(thta == 370)[0]
        isent = isent[0]
        uwndInstant = uwnd[i,:,:]
        vwndInstant = vwnd[i,:,:]
        uInstant = u[i,:,:,:]
        vInstant = v[i,:,:,:]
        uthta = pv.s2thta(lats,lons,plevs,kthta,uwndInstant,psfcInstant,uInstant,thta,pthta)
        vthta = pv.s2thta(lats,lons,plevs,kthta,vwndInstant,psfcInstant,vInstant,thta,pthta)
        ipvInstant = pv.sipv(lats,lons,kthta,thta,pthta,uthta,vthta,missingData)
        print(ipvInstant.shape)
        ipvPlot = ipvInstant[isent,:,:]
        print(ipvPlot.shape)
        uipvPlot = uthta[isent,:,:]
        vipvPlot = vthta[isent,:,:]
        ipvMeridional = pv.ddy(ipvPlot,lats,lons)
        plotIPV(lats,lons,ipvPlot,uipvPlot,vipvPlot,ipvMeridional,dates[i],temp=370)
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

    mean = (np.array(dates, dtype='datetime64[s]')
            .view('i8')
            .mean()
            .astype('datetime64[s]'))
    dateMean = mean.astype(datetime.datetime)
    plotIPV(lats,lons,ipvMean,uipvMean,vipvMean,ipvMeridional,dateMean,temp=370)
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
    line_c = ax1.contour(lons, lats, ipvPlot, levels=[0.7,1.5,2.0],
                         colors=['pink','red','black'],
                         transform=ccrs.PlateCarree())

    line_ipvgrad = ax1.contour(lons,lats,ipvMeridional,colors=['white'],transform=ccrs.PlateCarree())
    lons = lons[::3]
    lats = lats[::3]
    print(lons.shape,lats.shape,uipvPlot.shape,vipvPlot.shape)
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
    date = date.strftime('%Y-%m-%d-%H')
    print(date)
    isent = str(temp)
    plt.title('PV '+ isent+'K surface '+ date, fontsize=16)
    plt.savefig(date+'_spec'+isent+'K.png')
    plt.show()

            
main()
