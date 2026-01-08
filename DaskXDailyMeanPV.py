import warnings
from cartopy.io import DownloadWarning
warnings.filterwarnings("ignore", category=DownloadWarning)
import os
import numpy as np
from PV import potential_vorticity
from cdo import Cdo
from nco import Nco

import scipy.ndimage as ndimage
import math
import datetime
import xarray as xr
import re
import time
import cartopy
import dask
from dask.distributed import Client
from dask import delayed

cartopy_root = os.path.expanduser('~/.local/share/cartopy')


cartopy.config['data_dir'] = cartopy_root
cartopy.config['pre_existing_data_dir'] = cartopy_root


def main():

    #warnings.filterwarnings("ignore", category=DownloadWarning)
    cdo = Cdo()

    cdo.remapbil("myGridDef",input="surface_temp_2026_5_1_00Z.nc",output="surface_temp_5_1_2026.nc")
    cdo.remapbil("myGridDef",input="surface_uwnd_2026_5_1_00Z.nc",output="surface_uwnd_5_1_2026.nc")
    cdo.remapbil("myGridDef",input="surface_vwnd_2026_5_1_00Z.nc",output="surface_vwnd_5_1_2026.nc")


    startTime = time.time()

    
    tmp_file_list = [ file for file in os.listdir('.') if file.startswith("air_") ]
    uwnd_file_list = [ file for file in os.listdir('.') if file.startswith("uwnd_") ]
    vwnd_file_list = [ file for file in os.listdir('.') if file.startswith("vwnd_") ]

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


    cdo.merge(input=" ".join(('tmpFile.nc','uwndFile.nc','vwndFile.nc','surface_temp_5_1_2026.nc','pres_sfc_2026_5_1_00Z.nc','surface_uwnd_5_1_2026.nc','surface_vwnd_5_1_2026.nc')),output='pvFile.nc')

@delayed
def process_snapshot(i, lats, lons, plevs, tmp, tsfc, psfc, uwnd, vwnd, u, v, date, pv, missingData, target_theta):
    # Retrieve single snapshots from the 3D/4D arrays
    tmpI, tsfcI, psfcI = tmp[i], tsfc[i], psfc[i]
    
    # 1. Physics: Coordinate Transforms (thta defines the vertical levels)
    kthta, pthta, thta = pv.p2thta(lats, lons, plevs, tsfcI, psfcI, tmpI)
    
    # 2. Assignable Level Search (The superior argmin method)
    diffs = np.abs(thta - target_theta)
    isent = np.argmin(diffs)
    
    # 3. Isentropic Interpolation
    uthta = pv.s2thta(lats, lons, plevs, kthta, uwnd[i], psfcI, u[i], thta, pthta)
    vthta = pv.s2thta(lats, lons, plevs, kthta, vwnd[i], psfcI, v[i], thta, pthta)
    
    # 4. PV Calculation
    ipvInstant = pv.sipv2(lats, lons, kthta, thta, pthta, uthta, vthta, missingData)
    
    # Return ONLY the 2D slice for the chosen theta level
    return ipvInstant[isent]

    
def openFile():
    return xr.open_mfdataset("pvFile.nc", chunks={'time': 1})

def executeCalc(ds_pv,target_theta):
    # (Indented 4 spaces)
    pv = potential_vorticity()
    # ... prep coordinates and variables ...

    plevs = ds_pv.coords['level'].values * 100
    lats = ds_pv.coords['lat'].values
    lons = ds_pv.coords['lon'].values
    
    # Fix Longitude (0-360)
    lons = np.where(lons < 0, lons + 360, lons)
    
    dates = ds_pv.time.values.astype(str)
    dates = np.array([re.sub(r'\.+', ' ', re.sub('T', ' ', d)) for d in dates])

    date_mean = ds_pv.time.mean()
    date_mean = (date_mean.values)
    date_mean = np.datetime_as_string(date_mean)
    date_mean = re.sub('T0',' ',date_mean)
    date_mean = re.sub('\.[0]+',' ',date_mean)

    
    # Instantiate your physics engine consistently
    pv_engine = potential_vorticity() 
    missingData = -999.99
    
    # Extract values
    tmp, tsfc, psfc = ds_pv.air.values, ds_pv.air_2.values, ds_pv.pres.values
    uwnd, vwnd = ds_pv.uwnd_2.values, ds_pv.vwnd_2.values
    u, v = ds_pv.uwnd.values, ds_pv.vwnd.values

    
    tasks = []
    for i in range(tmp.shape[0]):
        task = process_snapshot(i, lats, lons, plevs, tmp, tsfc, psfc, 
                                uwnd, vwnd, u, v, dates[i], 
                                pv, -999.99, target_theta)
        tasks.append(task)
    
    # HEAVY MATH HAPPENS HERE
    results = dask.compute(*tasks)
    return date_mean,dates, results, lats, lons, pv


def plotIPV(lats, lons, ipv_snap, ipv_merid, date, temp):
    ax1 = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    clevs = np.arange(-11,11,1.0)
    shear_fill = ax1.contourf(lons,lats,ipv_snap,clevs,
                              transform=ccrs.PlateCarree(), cmap=plt.get_cmap('hsv'),
                              extend='both')
    line_c = ax1.contour(lons, lats, ipv_snap, levels=[1.5,2.0,3.0,4.0],
                         colors=['red','yellow','pink','white'],
                         transform=ccrs.PlateCarree())
    
    line_ipvgrad = ax1.contour(lons,lats,ipv_merid,colors=['white'],transform=ccrs.PlateCarree())
    lons = lons[::3]
    lats = lats[::3]
    #print(lons.shape,lats.shape,uipvPlot.shape,vipvPlot.shape)
    #ax1.quiver(lons,lats,uipvPlot,vipvPlot,transform=ccrs.PlateCarree())
    ax1.coastlines(resolution='50m',linewidth=0.5)
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
    isent = str(temp)
    plt.title('PV '+ isent+'K surface '+ date, fontsize=16)
    plt.savefig(date+'_spec'+isent+'K.png')
    plt.show()
    plt.close()

    
    
if __name__ == "__main__":
    # 1. Run CDO setup in the main process ONLY
    print("Running CDO setup...")
    main() 
    
    # 2. Start Dask client after setup is done
    client = Client(n_workers=3, threads_per_worker=1, memory_limit='8GB')
    print(f"Dask Dashboard is live at: {client.dashboard_link}")
    
    try:
        # 3. Load data and execute parallel calculation
        ds = openFile()
        date_mean,dates, results, lats, lons, pv = executeCalc(ds,360.)
        print("Calculation complete.")
    finally:
        # 4. Shut down workers
        client.close()
    if results:
        import cartopy.crs as ccrs
        from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
        from cartopy.util import add_cyclic_point
        import matplotlib as mpl
        mpl.rcParams['mathtext.default'] = 'regular'
        import matplotlib.pyplot as plt
        import cartopy
        for i, ipv_snap in enumerate(results):
            ipv_merid = pv.ddy(ipv_snap, lats, lons)
            # Assuming plotIPV is defined elsewhere or imported
            plotIPV(lats, lons, ipv_snap, ipv_merid, dates[i], temp=360)
            
    ipv_all = np.stack(results, axis=0)
    ipv_mean = np.mean(ipv_all, axis=0)
    
    # Grand Mean Plot
    ipv_mean_merid = pv.ddy(ipv_mean, lats, lons)
    plotIPV(lats, lons, ipv_mean, ipv_mean_merid, date_mean, temp=360)


    



