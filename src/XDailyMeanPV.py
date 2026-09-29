import warnings
from cartopy.io import DownloadWarning
warnings.filterwarnings("ignore", category=DownloadWarning)
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
import cartopy

from abc import ABC, abstractmethod

from coordinate_transformers import tests2thta,testp2thta,isobaric_to_isentropic_pressure
from coordinate_transformers import isobaric_to_isentropic_velocity

cartopy_root = os.path.expanduser('~/.local/share/cartopy')


cartopy.config['data_dir'] = cartopy_root
cartopy.config['pre_existing_data_dir'] = cartopy_root



class GridHandler(ABC):
    @abstractmethod
    def normalize(self, *args, **kwargs):
        pass

class XarrayGridHandler(GridHandler):
    """Strategy for Xarray objects: Leverages coordinate-aware sorting."""
    def normalize(self, obj):
        # 1. Flip Latitude to South-to-North (Ascending) if it arrives descending
        if obj.lat.values[0] > obj.lat.values[-1]:
            obj = obj.sortby("lat", ascending=True)
            
        # 2. Normalize Longitude to 0-360 boundaries and ensure it is monotonic
        obj = obj.assign_coords(lon=(obj.lon % 360.0))
        return obj.sortby("lon", ascending=True)

class NumpyGridHandler(GridHandler):
    """Strategy for raw NumPy arrays: Manually synchronizes lats, lons, and fields."""
    def normalize(self, lats, lons, *fields):
        # 1. Flip Lats/Fields to South-to-North if descending
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            fields = [np.flip(f, axis=0) for f in fields]
            
        # 2. Normalize Longitude to 0-360 boundaries
        lons = lons % 360.0
        
        # 3. Ensure monotonic longitude (West-to-East layout)
        if not np.all(np.diff(lons) > 0):
            idx = np.argsort(lons)
            lons = lons[idx]
            fields = [f[..., idx] for f in fields]
            
        return lats, lons, *fields


def main():

    #warnings.filterwarnings("ignore", category=DownloadWarning)

    cdo = Cdo()

    cdo.remapbil("myGridDef",input="surface_temp_2026_17_3_00Z.nc",output="surface_temp_17_3_2026.nc")
    cdo.remapbil("myGridDef",input="surface_uwnd_2026_17_3_00Z.nc",output="surface_uwnd_17_3_2026.nc")
    cdo.remapbil("myGridDef",input="surface_vwnd_2026_17_3_00Z.nc",output="surface_vwnd_17_3_2026.nc")


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


    cdo.merge(input=" ".join(('tmpFile.nc','uwndFile.nc','vwndFile.nc','surface_temp_17_3_2026.nc','pres_sfc_2026_17_3_00Z.nc','surface_uwnd_17_3_2026.nc','surface_vwnd_17_3_2026.nc')),output='pvFile.nc')

        
    # 4. Import the module (use the prefix, not the full .so filename)
    if __name__ == "__main__":
        #client = Client()
        ds_pv = openFile()
        executeCalc(ds_pv)
        stopTime = time.time()
        print(stopTime-startTime)
        
def openFile():
            
    ds_pv = xr.open_mfdataset("pvFile.nc",chunks={'time':'1'})
    print("here")
    print(ds_pv)
    print(ds_pv['uwnd'].data)
    return ds_pv
def executeCalc(ds_pv):
    """
    Core Numerical Coordinator.
    Maintains North-to-South fields for windspharm, but normalizes 
    spatial layers to South-to-North specifically for ddy diagnostics.
    """
    # 1. Enforce strict North-to-South alignment at the front gate for windspharm
    if ds_pv.lat.values[0] < ds_pv.lat.values[-1]:
        ds_pv = ds_pv.sortby("lat", ascending=False)
    
    levels = ds_pv.coords['level'].values
    plevs = np.asarray(levels, dtype=np.float64) * np.float64(100.0)
    
    # Extract coordinates in their native North-to-South orientation
    lats_n2s = ds_pv.coords['lat'].values
    lons_raw = ds_pv.coords['lon'].values
    dates    = ds_pv.time.values.astype(str)
    
    tmp  = ds_pv.air.values      
    tsfc = ds_pv.air_2.values    
    psfc = np.asarray(ds_pv.pres.values, dtype=np.float64)  
    uwnd = ds_pv.uwnd_2.values   
    vwnd = ds_pv.vwnd_2.values   
    u    = ds_pv.uwnd.values     
    v    = ds_pv.vwnd.values     
    
    pv_engine = potential_vorticity()
    numpy_handler = NumpyGridHandler()
    missingData = -999.99
    
    # Resolve your initial target surface index level bounds
    ret = isobaric_to_isentropic_pressure(tmp[0,:,:,:], plevs, tsfc[0,:,:], psfc[0,:,:])
    pthta_init, thta = ret[0], ret[1]
    isent = np.where(thta == 360.0)[0][0]
    
    num_timesteps = tmp.shape[0]
    for i in range(num_timesteps):
        print(f"\n--> PROCESSING TIMESTEP: {i+1} / {num_timesteps} | TIME: {dates[i]}")
        
        # Slices remain strictly North-to-South to prevent windspharm sign flips
        tmpInstant  = tmp[i, :, :, :]   
        tsfcInstant = tsfc[i, :, :]     
        psfcInstant = psfc[i, :, :]     
        uwndInstant = uwnd[i, :, :]     
        vwndInstant = vwnd[i, :, :]     
        uInstant    = u[i, :, :, :]     
        vInstant    = v[i, :, :, :]     
        
        pthta, thta = isobaric_to_isentropic_pressure(tmpInstant, plevs, tsfcInstant, psfcInstant)
        uthta = isobaric_to_isentropic_velocity(plevs, uInstant, pthta, psfcInstant, uwndInstant)
        vthta = isobaric_to_isentropic_velocity(plevs, vInstant, pthta, psfcInstant, vwndInstant)
        
        # Compute Ertel Potential Vorticity natively in North-to-South space
        ipvInstant = pv_engine.sipv2(lats_n2s, lons_raw, pthta.shape[0], thta, pthta, uthta, vthta, missingData)
        
        # 2. Extract the target 2D 360K surface layer matrices slice
        ipv_n2s = ipvInstant[isent, :, :]
        
        # 3. FIXED: Normalize the 2D layer to South-to-North *ONLY* for the finite difference code!
        lats_s2n, lons_s2n, ipv_s2n = numpy_handler.normalize(lats_n2s, lons_raw, ipv_n2s)
        
        # Compute your meridional sign transition gradients over a clean South-to-North space
        #ipvMeridional = pv_engine.ddy(ipv_s2n, lats_s2n, lons_s2n)
        import metpy.calc as mpcalc
        from metpy.units import units
        
        # 1. Purify missing data elements to prevent NaN mathematical errors in MetPy
        ipv_clean = np.where(ipv_s2n == missingData, np.nan, ipv_s2n)
        
        # 2. Attach required physical dimension units metadata tags 
        # Standard Potential Vorticity Units: m^2 s^-1 K kg^-1 (scaled by 1e6 already)
        ipv_with_units = ipv_clean * (units.m**2 / (units.s * units.kg * units.K))
        lats_with_units = lats_s2n * units.degrees
        
        # 3. Invoke MetPy's first_derivative utility
        # Specifying axis=0 targets your latitude rows sequence index natively
        # MetPy handles spherical delta-y distance mapping automatically!
        ipv_gradient_metpy = mpcalc.first_derivative(
            ipv_with_units, 
            axis=0, 
            x=lats_with_units
        )
        
        # 4. Extract raw numeric data arrays and restore missing data placeholders
        ipvMeridional = ipv_gradient_metpy.magnitude
        ipvMeridional = np.where(np.isnan(ipvMeridional), missingData, ipvMeridional)
        
        # 4. Dispatch the perfectly aligned South-to-North views directly to your canvas script
        plotIPV(lats_s2n, lons_s2n, ipv_s2n, ipvMeridional, dates[i], temp=360)

    
def plotIPV(lats,lons,ipvPlot,ipvMeridional,date,temp):

    np.set_printoptions(threshold=sys.maxsize)
    #print(uipvPlot)
    #print(vipvPlot)
    ax1 = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    clevs = np.arange(-11,11,1.0)
    shear_fill = ax1.contourf(lons,lats,ipvPlot,clevs,
                              transform=ccrs.PlateCarree(), cmap=plt.get_cmap('hsv'),
                              extend='both')
    line_c = ax1.contour(lons, lats, ipvPlot, levels=[1.5,2.0,3.0,4.0],
                         colors=['red','yellow','pink','white'],
                         transform=ccrs.PlateCarree())

    line_ipvgrad = ax1.contour(lons,lats,ipvMeridional,colors=['white'],transform=ccrs.PlateCarree())
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
    print(date)
    isent = str(temp)
    plt.title('PV '+ isent+'K surface '+ date, fontsize=16)
    plt.savefig(date+'_spec'+isent+'K.png')
    plt.show()

            
main()
