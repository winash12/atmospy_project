import xarray as xr
import pandas as pd

def final_core_merge(date_stamp="20260920"):
    cycles = ["00", "06", "12", "18"]
    snap_list = []

    for cyc in cycles:
        stamp = f"{date_stamp}{cyc}"
        print(f"--- Processing Cycle {cyc} ---")

        # 1. Open ONLY the Pressure Levels
        ds_pgb = xr.open_dataset(f"pgb.{stamp}.grb", engine="cfgrib", 
                                 backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
        
        print(f"Cycle {cyc} levels found: {len(ds_pgb.isobaricInhPa)}")
        ds_pgb = ds_pgb.rename({'t': 'air', 'u': 'uwnd', 'v': 'vwnd'})

        # 2. Open Surface Pressure and Temp
        ds_sfc_pt = xr.open_dataset(f"flx.{stamp}.grb", engine="cfgrib",
                                    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'surface'}})
        
        # 3. Open 10m Winds
        ds_sfc_uv = xr.open_dataset(f"flx.{stamp}.grb", engine="cfgrib",
                                    backend_kwargs={'filter_by_keys': {'typeOfLevel': 'heightAboveGround'}})

        # 4. Rename and Merge
        ds_pt = ds_sfc_pt.rename({'sp': 'pres', 't': 'air_2'})
        ds_uv = ds_sfc_uv.rename({'u10': 'uwnd_2', 'v10': 'vwnd_2'})

        combined = xr.merge([ds_pgb, ds_pt, ds_uv])
        combined = combined.expand_dims(time=[pd.to_datetime(stamp, format='%Y%m%d%H')])
        snap_list.append(combined)

    # 5. Final Assembly & Synchronization
    ds_final = xr.concat(snap_list, dim="time")
    
    # FIXED: Rename isobaric levels to 'level', and map latitude/longitude to match your 'lat'/'lon' requirements!
    ds_final = ds_final.rename({
        'isobaricInhPa': 'level',
        'latitude': 'lat',
        'longitude': 'lon'
    })
    
    # Sort levels descending (1000 hPa down to 0.4 hPa) as expected by windspharm/interpolation backends
    ds_final = ds_final.sortby("level", ascending=False)
    
    # FIXED: Changed output filename to 'pvFile.nc' to feed your driver loops seamlessly
    ds_final.to_netcdf("pvFile.nc")
    print("--- SUCCESS! All levels, coordinates, and variables merged into pvFile.nc ---")

if __name__ == "__main__":
    final_core_merge()
