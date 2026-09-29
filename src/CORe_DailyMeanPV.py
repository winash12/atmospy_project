# =============================================================================
#         PRAMANA VAYU (PV) CORE ENGINE - TIMELINE SIMULATION RUNNER
# =============================================================================
import os
import sys
import re
import time
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from abc import ABC, abstractmethod

# Core transformations facades and physics engines ingestion hooks
from coordinate_transformers import isobaric_to_isentropic_pressure, isobaric_to_isentropic_velocity
from PV import potential_vorticity

# =============================================================================
# 📂 SECTION 1: UPSTREAM GRID NORMALIZATION HANDLERS
# =============================================================================

class GridHandler(ABC):
    @abstractmethod
    def normalize(self, *args, **kwargs):
        pass

class NumpyGridHandler(GridHandler):
    """Strategy for raw NumPy arrays: Synchronizes fields to South-to-North layout."""
    def normalize(self, lats, lons, *fields):
        # 1. Flip Lats and 2D arrays to South-to-North if latitudes are descending
        if lats[0] > lats[-1]:
            lats = lats[::-1]
            fields = [np.flip(f, axis=0) for f in fields]
            
        # 2. Normalize Longitude coordinate boundaries to 0-360 degrees
        lons = lons % 360.0
        
        # 3. Ensure monotonic longitude layout (West-to-East)
        if not np.all(np.diff(lons) > 0):
            idx = np.argsort(lons)
            lons = lons[idx]
            fields = [f[..., idx] for f in fields]
            
        return lats, lons, *fields

# =============================================================================
# 📂 SECTION 2: PRODUCTION CARTOGRAPHY PLOTTING WINDOW
# =============================================================================

def plotIPV(lats, lons, ipvPlot, ipvMeridional, date_str, temp):
    """Renders high-resolution global filled contours on PlateCarree projections."""
    fig = plt.figure(figsize=(12, 7))
    ax1 = plt.axes(projection=ccrs.PlateCarree(central_longitude=180.0))
    
    # Establish colorbar contours boundaries (-11 to 11 PVU)
    clevs = np.arange(-11.0, 11.0, 1.0)
    
    # Render background PV field contours
    shear_fill = ax1.contourf(
        lons, lats, ipvPlot, clevs,
        transform=ccrs.PlateCarree(), 
        cmap=plt.get_cmap('hsv'),
        extend='both'
    )
    
    # Superimpose dynamic meteorological Tropopause bounds lines (1.5 to 4.0 PVU)
    ax1.contour(
        lons, lats, ipvPlot, 
        levels=[1.5, 2.0, 3.0, 4.0],
        colors=['red', 'yellow', 'pink', 'white'],
        transform=ccrs.PlateCarree(),
        linewidths=0.85
    )
    
    # Superimpose white meridional baroclinic sign changes guides lines cleanly
    if ipvMeridional is not None:
        try:
            ax1.contour(
                lons, lats, ipvMeridional, 
                colors=['white'], 
                transform=ccrs.PlateCarree(),
                linewidths=0.5,
                alpha=0.7
            )
        except Exception:
            pass
            
    # Add map features and tick labels styling parameters
    ax1.coastlines(resolution='50m', linewidth=0.5, color='gray')
    ax1.set_xticks([0, 60, 120, 180, 240, 300, 359.99], crs=ccrs.PlateCarree())
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    
    lon_formatter = LongitudeFormatter(zero_direction_label=True, number_format='.0f')
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)
    
    cbar = plt.colorbar(shear_fill, orientation='horizontal', pad=0.08, shrink=0.85)
    cbar.set_label('Isentropic Potential Vorticity [PVU = $10^{-6}$ m$^2$ s$^{-1}$ K kg$^{-1}$]')
    
    plt.title(f"PV {temp}K Surface | {date_str}", fontsize=14, weight='bold', pad=12)
    
    output_filename = f"{date_str}_spec{temp}K.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"--> [PLOT EXPORT] Saved production canvas frame asset to: {output_filename}")
    
    plt.show()
    plt.clf()
    plt.close(fig)

# =============================================================================
# 📂 SECTION 3: CALCULATION SCHEDULER & ITERATOR LOOP
# =============================================================================

def executeCalc(ds_pv):
    """
    Coordinates data extraction loops.
    Maintains North-to-South arrays for windspharm core math, 
    but shifts 2D matrices to South-to-North for ddy gradient calculations.
    """
    # 1. Enforce a clean North-to-South layout for incoming fields to satisfy windspharm constraints
    if ds_pv.lat.values[0] < ds_pv.lat.values[-1]:
        ds_pv = ds_pv.sortby("lat", ascending=False)
        
    levels = ds_pv.coords['level'].values
    plevs = np.asarray(levels, dtype=np.float64) * np.float64(100.0)
    
    # Store standard operational coordinates mapping vectors
    lats_n2s = ds_pv.coords['lat'].values
    lons_raw = ds_pv.coords['lon'].values
    dates    = ds_pv.time.values.astype(str)
    
    # Extract structural chunks variables arrays views
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
    
    # Loop continuously across every single available time step indices slot
    num_timesteps = tmp.shape[0]
    for i in range(num_timesteps):
        date_stamp = re.sub('T', ' ', dates[i])
        date_stamp = re.sub('\.[0]+', '', date_stamp)
        
        print(f"\n" + "="*80)
        print(f"--> RUNNING TRANSFORMATION PIPELINE TIMESTEP: {i+1} / {num_timesteps} | {date_stamp}")
        print(f"="*80)
        
        # Slices remain strictly North-to-South to lock down correct windspharm absolute vorticity signs
        tmpInstant  = tmp[i, :, :, :]   
        tsfcInstant = tsfc[i, :, :]     
        psfcInstant = psfc[i, :, :]     
        uwndInstant = uwnd[i, :, :]     
        vwndInstant = vwnd[i, :, :]     
        uInstant    = u[i, :, :, :]     
        vInstant    = v[i, :, :, :]     
        
        # FIXED: Correctly unpacks exactly 2 returned parameters from your updated facade structure!
        start_time = time.time()
        pthta, thta = isobaric_to_isentropic_pressure(tmpInstant, plevs, tsfcInstant, psfcInstant)
        stop_time = time.time()
        print(stop_time-start_time)
        # Locate target isentropic potential temperature coordinates row indices
        isent_indices = np.where(thta == 360.0)[0]
        if len(isent_indices) == 0:
            print(f"[WARNING] 360K level context missing on step index {i}. Skipping iteration.")
            continue
        isent = isent_indices[0]
        
        # Map velocity vector tracks natively
        start_time = time.time()
        uthta = isobaric_to_isentropic_velocity(plevs, uInstant, pthta, psfcInstant, uwndInstant)
        end_time = time.time()
        print(end_time-start_time)
        start_time = time.time()
        vthta = isobaric_to_isentropic_velocity(plevs, vInstant, pthta, psfcInstant, vwndInstant)
        end_time = time.time()
        print(end_time-start_time)
        sys.exit()
        # Compute absolute vorticity and Ertel Potential Vorticity fields
        ipvInstant = pv_engine.sipv2(lats_n2s, lons_raw, pthta.shape[0], thta, pthta, uthta, vthta, missingData)
        
        # Extract the target 2D horizontal 360K canvas fields layer slice
        ipv_n2s_slice = ipvInstant[isent, :, :]
        
        # 2. HYBRID NORMALIZATION: Normalize the 2D layer to South-to-North layout *ONLY* for the finite difference code!
        lats_s2n, lons_s2n, ipv_s2n = numpy_handler.normalize(lats_n2s, lons_raw, ipv_n2s_slice)
        
        # 3. Compute precise, stable meridional gradients over your South-to-North workspace coordinates
        ipvMeridional = pv_engine.ddy(ipv_s2n, lats_s2n, lons_s2n)
        
        # 4. Dispatch the perfectly aligned South-to-North matrices views directly to your cartopy plot function
        plotIPV(lats_s2n, lons_s2n, ipv_s2n, ipvMeridional, date_stamp, temp=360)

# =============================================================================
# 📂 SECTION 4: PIPELINE DRIVER HOOK
# =============================================================================

if __name__ == "__main__":
    print("--> Booting Pramana Vayu operational calculation engine loops...")
    if not os.path.exists("pvFile.nc"):
        print("[CRITICAL] Consolidated NetCDF production master asset 'pvFile.nc' not detected on disk.")
        sys.exit(1)
        
    ds_pv = xr.open_mfdataset("pvFile.nc", chunks={'time': '1'})
    executeCalc(ds_pv)
