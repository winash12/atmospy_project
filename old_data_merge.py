from cdo import Cdo 
import os
def main():

    #warnings.filterwarnings("ignore", category=DownloadWarning)
    cdo = Cdo()

    cdo.remapbil("myGridDef",input="surface_temp_2026_17_3_00Z.nc",output="surface_temp_17_3_2026.nc")
    cdo.remapbil("myGridDef",input="surface_uwnd_2026_17_3_00Z.nc",output="surface_uwnd_17_3_2026.nc")
    cdo.remapbil("myGridDef",input="surface_vwnd_2026_17_3_00Z.nc",output="surface_vwnd_17_3_2026.nc")


    
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

main()
