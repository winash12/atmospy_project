import sys
from datetime import date
import math
import re

class AtmosphericVariablesUtility:

    variableList = []
    pressureLevels = []
    specificHumidityPressureLevels = []
    variableLevelDictionary = {}

    def __init__(self):
        self.createDictionaryOfVariableAndLevel()

    def getVariableLevelDictionary(self):
        return self.variableLevelDictionary

    def createDictionaryOfVariableAndLevel(self):

        variableList =  self.listOfVariables()
        pressureLevels = self.getAllPressureLevels()
        specificHumidityPressureLevels = self.getSpecificHumidityPressureLevels()
        self.variableLevelDictionary = {"air":pressureLevels,
                                        "hgt":pressureLevels,
                                        "uwnd":pressureLevels,
                                       "vwnd":pressureLevels,
                                       "rhum":pressureLevels 
                                       }


        
    def listOfVariables(self):
        variableList = ["air","hgt","uwnd","vwnd","rhm"]
        return variableList

    def pressureLevelIndex(self,pressure):
 
        pressureLevels = ["1000","925","850","700","600","500","400","300","250","200","150","100","70","50","30","20","10"]
        pressureIndex = pressureLevels.index(pressure)
        return pressureIndex

    def getAllPressureLevels(self):
        pressureLevels = ["1000","925","850","700","600","500","400","300","250","200","150","100","70","50","30","20","10"]
        return pressureLevels

    def getSpecificHumidityPressureLevels(self):
        pressureLevels = ["1000","925","850","700","600","500","400","300"]
        return pressureLevels
        
        
class SpatialCoverage:


    def __init__(self):
        self.latList = []
        self.lonList = []
        self.gaussianLonList = []
        self.setMinMaxLatRange()
        self.setMinMaxLonRange()
        self.setMinMaxGaussianLonRange()
        
    def getLatIndex(self,latitude):
        return self.latList.index(latitude)

    def getLonIndex(self,longitude):
        return self.lonList.index(longitude)

    def getGaussianLonIndex(self,longitude):
        return self.gaussianLonList.index(longitude)
    
    def setMinMaxLatRange(self):
        c = 90
        for counter in range(0,73):
            self.latList.append(float(c))
            #print(self.latList[counter])
            c -= 2.5
        return
        
    def setMinMaxLonRange(self):
        z = 0
        for counter in range(0,144):
            self.lonList.append(float(z))
    #        print(self.lonList[counter])
            z += 2.5
        return 

    def setMinMaxGaussianLonRange(self):
        z = 0
        for counter in range(0,192):
            self.gaussianLonList.append(float(z))
            #print(self.gaussianLonList[counter])
            z += 1.875
        return 

    
class TemporalCoverage:

    def getTimeIndex(self,year,month,day,time):

        date_end = date(year,month,day)
        date_start = date(year,1,1)
        days = (date_end - date_start).days
        #print("number of days is " + str(days))
        timeOfDay = ["00","06","12","18"]
        period =  timeOfDay.index(time)
        if (days == 0):
            timeIndex = int(period)
        else:
            daysIndex = days*4
            timeIndex = daysIndex + int(period)
        return timeIndex


class ConstraintExpressionGenerator:

    def __init__(self,year1,month1,day1,time1,year2,month2,day2,time2,
                 minLat,maxLat,minLon,maxLon):
        
        self.url = ""
        self.urlReanalysis = ""
        self.urlSurface = ""
        self.year1 = year1
        self.month1 = month1
        self.day1 = day1
        self.time1 = time1
        self.year2 = year2
        self.month2 = month2
        self.day2 = day2
        self.time2 = time2
        self.minLat = minLat
        self.maxLat = maxLat
        self.minLon = minLon
        self.maxLon = maxLon
        tempCoverage = TemporalCoverage()
        self.timeConstraint1 = tempCoverage.getTimeIndex(year1,month1,day1,time1)
        self.timeConstraint2 = tempCoverage.getTimeIndex(year2,month2,day2,time2)
        self.atmosUtility = AtmosphericVariablesUtility()
        self.spatialCoverage = SpatialCoverage()
        
        
    def getConstraintExpression(self,urlReanalysis,atmosphericVariable,pressureLevel):

        print(urlReanalysis)
        levelConstraint = str(self.atmosUtility.pressureLevelIndex(pressureLevel))

        latConstraint = self.getLatConstraint(self.minLat,self.maxLat)
        
        lonConstraint = self.getLonConstraint(self.minLon,self.maxLon)

        urlVariable1 =  atmosphericVariable + "." + str(self.year1)+"."+"nc?"
        urlVariable2 =  atmosphericVariable
        urlTimeIndex =  "["+str(self.timeConstraint1) + ":" + str(self.timeConstraint2) + "]"
        urlLevel =      "[" + levelConstraint + "]"
        urlLat =         latConstraint
        urlLon =         lonConstraint
        urlVariable3 = ",time[" + str(self.timeConstraint1) + ":" + str(self.timeConstraint2) + "]"
        urlVariable4 = ",level[" + levelConstraint + "]"
        
        urlVariable5 = ",lat" + latConstraint 

        urlVariable6 = ",lon" + lonConstraint  

        self.url = urlReanalysis + urlVariable1 + urlVariable2 + urlTimeIndex + urlLevel + urlLat + urlLon + urlVariable3 + urlVariable4 + urlVariable5 + urlVariable6
        return self.url
    

    def getConstraintExpressionForSurfacePressure(self):

        urlMSLP = "http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis/surface/"
        urlVariable1 =  "pres.sfc" + "." + str(self.year1)+"."+"nc?"
        urlTimeIndex =  "["+str(self.timeConstraint1) + ":" + str(self.timeConstraint2) + "]"
        latConstraint = self.getLatConstraint(self.minLat,self.maxLat)
        lonConstraint = self.getLonConstraint(self.minLon,self.maxLon)
        urlVariable2 = "pres" + urlTimeIndex + latConstraint+lonConstraint
        urlVariable3 = ",time[" + str(self.timeConstraint1) + ":" + str(self.timeConstraint2) + "]"
        urlVariable4 = ",lat" + latConstraint 

        urlVariable5 = ",lon" + lonConstraint  
        urlVariable = urlVariable1 + urlVariable2 + urlVariable3  + urlVariable4 + urlVariable5
        self.urlSurface = urlMSLP + urlVariable
        print(self.urlSurface)
        return self.urlSurface

    def getConstraintExpressionForSurfaceFlux(self,surfaceFluxVariable):

        print(surfaceFluxVariable)
        urlFlux = "http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep.reanalysis/surface_gauss/"
        surfaceFluxV = surfaceFluxVariable + "." + "gauss"
        urlVariable1 =  surfaceFluxV + "." + str(self.year1)+"."+"nc?"
        urlTimeIndex =  "["+str(self.timeConstraint1) + ":" + str(self.timeConstraint2) + "]"
        latConstraint = self.getGaussianLatConstraint()
        lonConstraint = self.getGaussianLonConstraint()
        surfaceFluxV1 = re.sub(r'\.[0-9]+[a-z]',"",surfaceFluxVariable)
        print(surfaceFluxV1)
        urlVariable2 =  surfaceFluxV1 + urlTimeIndex + latConstraint+lonConstraint
        urlVariable3 = ",time[" + str(self.timeConstraint1) + ":" + str(self.timeConstraint2) + "]"
        urlVariable4 = ",lat" + latConstraint 
        
        urlVariable5 = ",lon" + lonConstraint  
        urlVariable = urlVariable1 + urlVariable2 + urlVariable3  + urlVariable4 + urlVariable5
        self.urlSurface = urlFlux + urlVariable
        print(self.urlSurface)
        return self.urlSurface

    def getGaussianLatConstraint(self):
        maxLatIndex = 0
        minLatIndex = 93
        self.latConstraint = "[" + str(maxLatIndex) + ":" + str(minLatIndex) + "]"
        return self.latConstraint

    def getLatConstraint(self,minLat,maxLat):

        minLat = math.floor(minLat/2.5) * 2.5
        try:
            minLatIndex = self.spatialCoverage.getLatIndex(minLat)
        except:
            pass
            
        maxLat = math.ceil(maxLat/2.5)*2.5
        try:
            maxLatIndex = self.spatialCoverage.getLatIndex(maxLat)
        except:
            pass
        if (maxLatIndex >  minLatIndex):
            #Adjust for southern hemisphere latitudes as well
            #By switching
            #Always go from North to South(Indices must increase)
            copyV = minLatIndex
            minLatIndex = maxLatIndex
            maxLatIndex = copyV
        self.latConstraint = "[" + str(maxLatIndex) + ":" + str(minLatIndex) + "]"
        return self.latConstraint

    def getGaussianLonConstraint(self):
        minLon = 0.0
        maxLon = 358.125
        minLonIndex = self.spatialCoverage.getGaussianLonIndex(minLon)
        maxLonIndex = self.spatialCoverage.getGaussianLonIndex(maxLon)
        if (minLonIndex > maxLonIndex):
            copyV = minLonIndex
            minLonIndex = maxLonIndex
            maxLonIndex = copyV
        
        self.lonConstraint = "[" + str(minLonIndex) + ":" + str(maxLonIndex) + "]"
        return self.lonConstraint

    
    def getLonConstraint(self,minLon,maxLon):
        try:
            minLonIndex = self.spatialCoverage.getLonIndex(minLon)
        except:
            pass
        try:
            maxLonIndex = self.spatialCoverage.getLonIndex(maxLon)
        except:
            pass
        if (minLonIndex > maxLonIndex):
            copyV = minLonIndex
            minLonIndex = maxLonIndex
            maxLonIndex = copyV
        
        self.lonConstraint = "[" + str(minLonIndex) + ":" + str(maxLonIndex) + "]"
        return self.lonConstraint
    
