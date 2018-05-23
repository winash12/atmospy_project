#!/usr/bin/python3.5
import sys,os
import numpy as np
from netCDF4 import Dataset,num2date
from PV import potential_vorticity
from cdo import Cdo
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.util import add_cyclic_point
import matplotlib as mpl
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt


def main():
    pv = potential_vorticity()
    
    pres = ["100000","92500","85000","70000","60000","50000","40000","30000","25000","20000","15000","10000","7000","5000","3000","2000","1000"]
    tmp_file_list = [ file for file in os.listdir('.') if file.startswith("air") ]
    uwnd_file_list = [ file for file in os.listdir('.') if file.startswith("uwnd") ]
    vwnd_file_list = [ file for file in os.listdir('.') if file.startswith("vwnd") ]

    #Start
    tmpFilePressureLevelDictionary={}
    for file in tmp_file_list:
        pressureLevel = int(file.split("_")[1])
        tmpFilePressureLevelDictionary[pressureLevel] = file
    args = []
    args1 = []
    missing  = 0
    for key in  sorted(tmpFilePressureLevelDictionary,reverse=True):
        file = tmpFilePressureLevelDictionary[key]
        nc_tempFile = Dataset(file,'r')
        lats = nc_tempFile.variables['lat'][:]  # extract/copy the data
        lons = nc_tempFile.variables['lon'][:]
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
        temp = np.reshape(temp,(nj,ni,1))
        #temp_theta = nc_tempFile.variables['air'][0,:]*units(nc_tempFile.variables['air'].units)
        # Reanalysis data is oriented N-S. So reverse the array
        temp = temp[::-1,:,:]
        args.append(temp)
        #args1.append(temp_theta)
    tmp = np.concatenate(args,axis=2)
    #tmp_theta = np.concatenate(args1,axis=0)

    #print(missingData)
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
        uwnd = np.reshape(uwnd,(nj,ni,1))
        #uwnd_theta = nc_uwndFile.variables['uwnd'][0,:]*units(nc_uwndFile.variables['uwnd'].units)
        uwnd = uwnd[::-1,:,:]
        args.append(uwnd)
        #args1.append(uwnd_theta)
    u = np.concatenate(args,axis=2)
    #u_theta = np.concatenate(args1,axis=0)
    #Done


        
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
        vwnd = np.reshape(vwnd,(nj,ni,1))
        vwnd = vwnd[::-1,:,:]
        #vwnd_theta = nc_vwndFile.variables['vwnd'][0,:]*units(nc_vwndFile.variables['vwnd'].units)
        args.append(vwnd)
        #args1.append(vwnd_theta)
    v = np.concatenate(args,axis=2)
    #v_theta = np.concatenate(args1,axis=0)
    #Done

    tmp_file = "surface_temp_20_5_2018.nc"
    nc_surfTFile = Dataset(tmp_file,'r')
    tsfc = nc_surfTFile.variables['air'][:]
    tsfc = np.reshape(tsfc,(nj,ni))
    tsfc = tsfc[::-1,:]
    
    pres_file = "pres_sfc_2018_20_5_00Z.nc"
    nc_presFile = Dataset(pres_file,'r')
    psfc = nc_presFile.variables['pres'][:]
    psfc = np.reshape(psfc,(nj,ni))
    psfc = psfc[::-1,:]

    uwnd_10m_file = "surface_uwnd_20_5_2018.nc"
    nc_uwndFile = Dataset(uwnd_10m_file,'r')
    uwnd = nc_uwndFile.variables['uwnd'][:]
    uwnd = np.reshape(uwnd,(nj,ni))
    uwnd = uwnd[::-1,:]
    
    vwnd_10m_file = "surface_vwnd_20_5_2018.nc"
    nc_vwndFile = Dataset(vwnd_10m_file,'r')
    vwnd = nc_vwndFile.variables['vwnd'][:]
    vwnd = np.reshape(vwnd,(nj,ni))
    vwnd = vwnd[::-1,:]
    
    plevs = np.array(pres)
    plevs = plevs.astype(np.float)
    ret = pv.p2thta(lats,lons,plevs,tsfc,psfc,tmp)
    kthta,pthta,thta = ret
    uthta = pv.s2thta(lats,lons,plevs,kthta,uwnd,psfc,u,thta,pthta)
    vthta = pv.s2thta(lats,lons,plevs,kthta,vwnd,psfc,v,thta,pthta)
    ipv = pv.ipv(lats,lons,kthta,thta,pthta,uthta,vthta,missingData)

    for i in range (0,len(lons)):
        if (lons[i] < 0.0):
            lons[i] += 360.0
    ipvPlot = np.empty((nj,ni))
    ipvPlot = ipv[2,:,:]*1e6
    print(max(ipvPlot.flatten()))
    print(min(ipvPlot.flatten()))
    sys.exit()
    ax1 = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    clevs = np.arange(-18,18,1)
    shear_fill = ax1.contourf(lons,lats,ipvPlot,clevs,
                              transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r,
                                      linewidth=(10,),extend='both')
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
    plt.title('Isentropic PV On The 380 K surface T vs ln P interpolation', fontsize=16)
    plt.savefig('pv.png')
    plt.show()
    sys.exit()

    #pvor = np.empty((nj,ni))

    #tmp_theta = tmp_theta * units.kelvin
    #isentlevs = [265,275,285,300,315,320,330,350,350,370,395,430,475,530,600,700] * units.kelvin
    #isent_anal = performIsentropicInterpolation(isentlevs,plevs,tmp_theta,u_theta,v_theta)
    #isentprs,isenttmp,isentu,isentv=isent_anal
    #isentprs = np.reshape(isentprs,(nj,ni,16))
    #isenttmp = np.reshape(isenttmp,(nj,ni,16))
    #isentu = np.reshape(isentu,(nj,ni,16))
    #isentv = np.reshape(isentv,(nj,ni,16))

    for i in range (0,len(lons)):
        if (lons[i] < 0.0):
            lons[i] += 360.0
    for k in range(0,16):
            #pvor = pv.pvonp(ni,nj,lats,lons,pres[k],pres[k+1],pres[k],tmp[:,:,k],tmp[:,:,k+1],tmp[:,:,k],u[:,:,k],u[:,:,k+1],u[:,:,k],v[:,:,k],v[:,:,k+1],v[:,:,k],missingData)
        #elif (k == 16):
        #    pvor = pv.pvonp(ni,nj,lats,lons,pres[k],pres[k-1],pres[k],tmp[:,:,k],tmp[:,:,k-1],tmp[:,:,k],u[:,:,k],u[:,:,k-1],u[:,:,k],v[:,:,k],v[:,:,k-1],v[:,:,k],missingData)
         #   print("NO")
        #else:
        if (k ==9):
            #pvor = pv.pvonp(ni,nj,lats,lons,pres[k],pres[k+1],pres[k-1],tmp[:,:,k],tmp[:,:,k+1],tmp[:,:,k-1],u[:,:,k],u[:,:,k+1],u[:,:,k-1],v[:,:,k],v[:,:,k+1],v[:,:,k-1],missingData)
            pvor = pv.pvlayr(ni,nj,lats,lons,pres[k],pres[k+1],pres[k-1],tmp[:,:,k+1],tmp[:,:,k-1],u[:,:,k+1],u[:,:,k-1],v[:,:,k+1],v[:,:,k-1])
            #pvor = pv.pv_rossby(ni,nj,lats,lons,pres[k],pres[k+1],pres[k-1],tmp[:,:,k],tmp[:,:,k+1],tmp[:,:,k-1],u[:,:,k],u[:,:,k+1],u[:,:,k-1],v[:,:,k],v[:,:,k+1],v[:,:,k-1],missingData)
            #pvor = pv.pv_isobaric_bluestein(ni,nj,lats,lons,pres[k],pres[k+1],pres[k-1],tmp[:,:,k],tmp[:,:,k+1],tmp[:,:,k-1],u[:,:,k],u[:,:,k+1],u[:,:,k-1],v[:,:,k],v[:,:,k+1],v[:,:,k-1],missingData)
            #pvor = pv.pvonthetalayer(lats,lons,isentprs[:,:,k+1],isentprs[:,:,k-1],isentlevs[k+1],isentlevs[k-1],u[:,:,k+1],u[:,:,k-1],v[:,:,k+1],v[:,:,k-1],missingData)
            #print(max(pvor.flatten()))
            #print(min(pvor.flatten()))
            #print
            ax1 = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
            clevs = np.arange(-10,10,1)
            shear_fill = ax1.contourf(lons,lats,pvor,clevs,
                                      transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r,
                                      linewidth=(10,),extend='both')
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
            plt.title('Isobaric Potential Vorticity On The 200 hPa surface', fontsize=16)
            plt.savefig('pv.png')
            plt.show()
                
            
main()
