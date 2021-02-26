#!/usr/bin/python3.5

import sys 

import configparser

import subprocess

import shlex

import numpy

from datadownloader import AtmosphericVariablesUtility,ConstraintExpressionGenerator

def executeDownload(atmosVariables,pressureLevels):


    for idx,val in enumerate(atmosVariables):

        atmosphericVariable = val
        
        if isAtmosphericVariable(atmosphericVariable):
        
            for idx,val in enumerate(pressureLevels):
                
                pressureLevel = val
                
                if isPressureLevel(atmosphericVariable,pressureLevel):
                    #print(atmosphericVariable)
                    #print(pressureLevel)
                    url=    createConstraintExpression(atmosphericVariable,
                                                       pressureLevel)
                    
                    print(url)
                    outputFileName = createOutputFileName(atmosphericVariable,
                                                          pressureLevel)

                    print(outputFileName)

                    command_list = shlex.split('nccopy -k 4 '+url+' '+ outputFileName)

                    subprocess.check_call(command_list)

def createOutputFileName(atmosphericVariable,pressureLevel):
    outputFileName = atmosphericVariable + "_" + pressureLevel + "_" + str(year1) + "_" + str(day1) + "_" + str(month1) + "_" + str(time1) + "Z" + ".nc"
    return outputFileName
                    

def createConstraintExpression(atmosphericVariable,pressureLevel):
    if atmosphericVariable == "shum":
        url = urlReanalysis
    else :
        url = urlReanalysis2
    return constraintExpressionGenerator.getConstraintExpression(url,atmosphericVariable,pressureLevel)

def isAtmosphericVariable(atmosphericVariable):
    if atmosphericVariable in variableLevelDictionary:
        return True
    else:
        return False

def isPressureLevel(atmosphericVariable,pressureLevel):
    list = variableLevelDictionary.get(atmosphericVariable)
    if pressureLevel in list:
        return True
    else:
        return False
    

def executeSurfacePressureDownload():
    url=    createConstraintExpressionForSurfacePressure()
    print(url)
    outputFile = "pres_sfc" + "_" + str(year1) + "_" + str(day1) + "_" + str(month1) + "_" + str(time1) + "Z" + ".nc"
    print(outputFile)
    command_list = shlex.split('nccopy -k 4 ' + url + ' '+ outputFile)
    subprocess.check_call(command_list)

surfaceVarDic = {}
surfaceVarDic["air.2m"]="surface_temp"
surfaceVarDic["uwnd.10m"]="surface_uwnd"
surfaceVarDic["vwnd.10m"]="surface_vwnd"

def executeSurfaceFluxDownload(surfaceFluxVariables):

    for val in surfaceFluxVariables:
        surfaceFluxVariable = val
        url=    createConstraintExpressionForSurfaceFlux(surfaceFluxVariable)
        #print(url)
        surfaceFluxVariable = surfaceVarDic[val]
        outputFile = surfaceFluxVariable + "_" + str(year1) + "_" + str(day1) + "_" + str(month1) + "_" + str(time1) + "Z" + ".nc"
        print(outputFile)
        command_list = shlex.split('nccopy -k 4 ' + url + ' '+ outputFile)
        subprocess.check_call(command_list)

def createConstraintExpressionForSurfacePressure():
    
    return constraintExpressionGenerator.getConstraintExpressionForSurfacePressure()        

def createConstraintExpressionForSurfaceFlux(surfaceFluxVariable):
    
    return constraintExpressionGenerator.getConstraintExpressionForSurfaceFlux(surfaceFluxVariable)        


def main():

    config = configparser.ConfigParser()
    config.read('noaa.cfg')
    atmosVariableUtility = AtmosphericVariablesUtility()
    global variableLevelDictionary,constraintExpressionGenerator
    global year1,month1,day1,time1,urlReanalysis2,urlReanalysis
    variableLevelDictionary = atmosVariableUtility.getVariableLevelDictionary()
    
    urlReanalysis2 = config.get('Key Values','urlReanalysis')
    urlReanalysis = config.get('Key Values','urlReanalysis')
    year1 = int(config.get('Key Values','year1'))
    month1 = int(config.get('Key Values','month1'))
    day1  = int(config.get('Key Values','day1'))
    time1 =  config.get('Key Values','time1')
    year2 = int(config.get('Key Values','year2'))
    month2 = int(config.get('Key Values','month2'))
    day2  = int(config.get('Key Values','day2'))
    time2 =  config.get('Key Values','time2')
    pressure = config.get('Key Values','pressureLevels')
    pressureLevels = pressure.split(",")
    #print(pressureLevels)
    atmos = config.get('Key Values','atmosVariables')
    atmosVariables = atmos.split(",")
    surfaceVariable = config.get('Key Values','surfaceVariables')
    surfaceFlux = config.get('Key Values','surfaceFluxVariables')
    surfaceFluxVariables = surfaceFlux.split(",")
    minLat = float(config.get('Key Values','minLat'))
    maxLat = float(config.get('Key Values','maxLat'))
    minLon = float(config.get('Key Values','minLon'))
    maxLon = float(config.get('Key Values','maxLon'))

    # Convert -180/180 to 0 through 360
    minLon = numpy.mod(minLon,360)
    maxLon = numpy.mod(maxLon,360)

    constraintExpressionGenerator = ConstraintExpressionGenerator(year1,
                                                                  month1,
                                                                  day1,
                                                                  time1,
                                                                  year2,
                                                                  month2,
                                                                  day2,
                                                                  time2,
                                                                  minLat,
                                                                  maxLat,
                                                                  minLon,
                                                                  maxLon)

    executeDownload(atmosVariables,pressureLevels)

    executeSurfacePressureDownload()

    executeSurfaceFluxDownload(surfaceFluxVariables)

main() # Call the main function

