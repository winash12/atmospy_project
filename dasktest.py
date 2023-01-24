import numpy as np
import dask
from dask import delayed
from dask.distributed import Client
import time
def main():
    
    if __name__ == "__main__":

        client = Client()
        tmp,pres = setUpData()
        startTime = time.time()
        executeCalc(tmp,pres)
        stopTime = time.time()
        print(stopTime-startTime)
    
def setUpData():
    
    temperature = 273 + 20 * np.random.random([4,17,73,144])
    pres = ["1000","925","850","700","600","500","400","300","250","200","150","100","70","50","30","20","10"]
    level = np.array(pres)
    level = level.astype(float)*100
    return tmp,level

def executeCalc(tmp,pres):
    potempList = []
    for i in range (0,tmp.shape[0]):
        tmpInstant = tmp[i,:,:,:]
        potempList.append(delayed(pot)(tmpInstant,pres))
    results = dask.compute(potempList,scheduler='processes',num_workers=4)
                          
def pot(tmp,pres):
    potemp = np.zeros((17,73,144))
    potemp = tmp * (100000./pres[:,None,None])
    return potemp
    
main()
