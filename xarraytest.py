import xarray as xr
import numpy as np
import datetime
import sys

def main():
    
    temperature = 273 + 20 * np.random.random([4,17,73,144])
    lat = np.linspace(-90.,90.,73)
    lon = np.linspace(0.,360. ,144,endpoint=False)
    pres = ["1000","925","850","700","600","500","400","300","250","200","150","100","70","50","30","20","10"]
    level = np.array(pres)
    level = level.astype(float)
    time = np.empty((4))
    time[0] = np.datetime64(datetime.datetime(2023,1,14,0))
    time[1] = np.datetime64(datetime.datetime(2023,1,14,6))
    time[2] = np.datetime64(datetime.datetime(2023,1,14,12))
    time[3] = np.datetime64(datetime.datetime(2023,1,14,18))
    temp = xr.DataArray(temperature,coords=[time,level,lat,lon],dims=['time', 'level','lat', 'lon'])
    ds = xr.Dataset(data_vars={'temperature':temp})
    ds.to_netcdf('testxarray.nc')
    ds1 = xr.open_dataset('testxarray.nc',chunks={'time':1})
    print(ds1)
    tmp = ds.temperature


main()
