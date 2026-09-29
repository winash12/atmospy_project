import warnings
from cartopy.io import DownloadWarning
warnings.filterwarnings("ignore", category=DownloadWarning)
import os
import numpy as np
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
import cartopy
import dask
from dask.distributed import Client
from dask import delayed
import sys
cartopy_root = os.path.expanduser('~/.local/share/cartopy')


cartopy.config['data_dir'] = cartopy_root
cartopy.config['pre_existing_data_dir'] = cartopy_root


def main():

    #warnings.filterwarnings("ignore", category=DownloadWarning)
    print("starting")

@delayed
def process_snapshot(i, lats, lons, plevs, tmp, tsfc, psfc, uwnd, vwnd, u, v, date, pv, missingData, target_theta=360.0):
    """
    Asynchronous Dask Worker Node.
    Leverages optimized public facade backends to achieve maximum concurrency 
    and drops the GIL instantly when invoking compiled C++ execution tracks.
    """
    # 1. Isolate individual 3D/2D grid fields snapshots for this time index slot
    tmpI  = tmp[i]   # Shape: (Levels, Lat, Lon)
    tsfcI = tsfc[i]  # Shape: (Lat, Lon)
    psfcI = psfc[i]  # Shape: (Lat, Lon)
    
    # 2. FIXED: Pull Pressure Transformations straight from your public facade module!
    # Returns: (pthta, thta) under your synchronized 2-element tuple specification pass
    pthta, thta = isobaric_to_isentropic_pressure(tmpI, plevs, tsfcI, psfcI)
    
    # 3. Locate Target Isentropic Layer (The highly robust argmin approach)
    diffs = np.abs(thta - target_theta)
    isent = np.argmin(diffs)
    
    # 4. FIXED: Route wind components straight through your lightning-fast facade entries!
    # Bypasses the Python interpreter loop bottleneck by hitting the C++ Shared Object Kernels natively
    uthta = isobaric_to_isentropic_velocity(plevs, u[i], pthta, psfcI, uwnd[i])
    vthta = isobaric_to_isentropic_velocity(plevs, v[i], pthta, psfcI, vwnd[i])
    
    # 5. Execute Ertel Potential Vorticity (sipv2 handles its internal vertical check layers)
    ipvInstant = pv.sipv2(lats, lons, pthta.shape[0], thta, pthta, uthta, vthta, missingData)
    
    # Return ONLY the high-resolution 2D layer slice for your target 360K surface canvas map
    return ipvInstant[isent]

def openFile():
    return xr.open_mfdataset("pvFile.nc", chunks={'time': 1})

def executeCalc(ds_pv):
    # (Indented 4 spaces)
    pv = potential_vorticity()
    # ... prep coordinates and variables ...
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
    missingData = -999.99
    
    # Extract values
    tmp, tsfc, psfc = ds_pv.air.values, ds_pv.air_2.values, ds_pv.pres.values
    uwnd, vwnd = ds_pv.uwnd_2.values, ds_pv.vwnd_2.values
    u, v = ds_pv.uwnd.values, ds_pv.vwnd.values

    
    tasks = []
    for i in range(tmp.shape[0]):
        task = process_snapshot(i, lats, lons, plevs, tmp, tsfc, psfc, 
                                uwnd, vwnd, u, v, dates[i], 
                                pv, -999.99, target_theta=360)
        tasks.append(task)
    
    # HEAVY MATH HAPPENS HERE
    results = dask.compute(*tasks)
    for i, ipv_snap in enumerate(results):
        ipv_merid = pv.ddy(ipv_snap, lats, lons)
        # Assuming plotIPV is defined elsewhere or imported
        plotIPV(lats, lons, ipv_snap, ipv_merid, dates[i], temp=360)
        
    ipv_all = np.stack(results, axis=0)
    ipvMean = np.mean(ipv_all, axis=0)
    
    # Grand Mean Plot
    ipv_mean_merid = pv.ddy(ipvMean, lats, lons)
    plotIPV(lats, lons, ipvMean, ipv_mean_merid, "Grand Mean", temp=360)

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
    client = Client(n_workers=3, threads_per_worker=1, memory_limit='8GB')
    try:
        ds = openFile()
        final_mean = executeCalc(ds)
    finally:
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
