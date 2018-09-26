
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:54:40 2018

@author: amal
"""

# noise introduction and removal-check type of noise in data - noise removal packages--check 
# work more on input data generation
# increase intensity of noise 
# timestamp based data work on that
# rare input variable not functioning
# make everything gaussian 
# confidence check for every interval check if the result changes, if not we can confirm to sample data

#steps for input
# generate data through gaussian-numpy
# calculate the dependent variables using data
# introduce the noise to data
# missing values of input

import datetime
import json
import math
import os
#from time import sleep
#import os
import sys
import pandas as panda
from sklearn.decomposition import PCA
import numpy as np
import itertools
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from scipy.signal import savgol_filter
import scipy.fftpack
#
#N = 100
#x = np.linspace(0,2*np.pi,N)
#y = np.sin(x) + np.random.random(N) * 0.2
#
#w = scipy.fftpack.rfft(y)
#f = scipy.fftpack.rfftfreq(N, x[1]-x[0])
#spectrum = w**2
#
#cutoff_idx = spectrum < (spectrum.max()/5)
#w2 = w.copy()
#w2[cutoff_idx] = 0
#
#y2 = scipy.fftpack.irfft(w2)

class dataAna:

    def __init__(self,filepath,graphFilePath):
        self.filepath=filepath
        self.graphFilePath=graphFilePath
        self.distanceBetweenSensors=[]
        
    def readInput(self):
        #reads input data
        #interpolates the missing values and drop not available values
        inputData=panda.read_csv(self.filepath)
        inputData=inputData.interpolate(method='nearest')
        inputData.dropna(inplace=True)
        inputData.reset_index(drop=True,inplace=True)
        return inputData
        
    def pointsDistance(self,x1,y1,x2,y2):
        # calculates the eucledian distance between input distance
        a=abs(math.degrees(math.atan(y2/x2)))
        b=abs(math.degrees(math.atan(y1/x1)))
        print(a,b)
        return abs(a-b)
        
    def processGraph(self):
        V=set()
        with open(self.graphFilePath,"r")  as jsonFile:
            graph=json.load(jsonFile)
            for vertex in graph:
                V.add(vertex)
                for near_node in graph[vertex]:
                    V.add(near_node)
                
        jsonFile.close()
        self.noOfVertex=len(V)
        self.vertexMap={}
        i=0
        for vertex in V:
            vertexList=[10000 for x in V]
            self.vertexMap[vertex]=i
            self.distanceBetweenSensors.append(vertexList)
            self.distanceBetweenSensors[i][i]=0
            i=i+1        
        
        for vertex in graph:
            for nearNode in graph[vertex]:
                i=self.vertexMap[vertex]
                j=self.vertexMap[nearNode]
                self.distanceBetweenSensors[i][j]=graph[vertex][nearNode]
                self.distanceBetweenSensors[j][i]=graph[vertex][nearNode]
        
        for k in range(0,self.noOfVertex):
            for i in range(0,self.noOfVertex):
                for j in range(0,self.noOfVertex):
                    self.distanceBetweenSensors[i][j]=min(self.distanceBetweenSensors[i][j],self.distanceBetweenSensors[i][k]+self.distanceBetweenSensors[k][j])
    
    def calculateCloselyCorrealtedVariables(self,averageDistance,distanceMatrix):
        #the average distance and distance between the variable projection with respect to pricipal components is input to the function
        #the variables close to each other in principal component space have similiar characterstics 
        #this principle can be used to identify potential relation among variables
        #this method discovers closely associated variables which can be analyzed further
        closeVariables={}
        for x in distanceMatrix.keys():
            if(distanceMatrix[x]<=40 or distanceMatrix[x]>=320):
                variables=x.split('&&')
                if(variables[0] not in closeVariables):
                    closeVariables[variables[0]]=[]
                    closeVariables[variables[0]].append(variables[1])
                else:
                    closeVariables[variables[0]].append(variables[1])
                if(variables[1] not in closeVariables):
                    closeVariables[variables[1]]=[]
                    closeVariables[variables[1]].append(variables[0])
                else:
                    closeVariables[variables[1]].append(variables[0])
        return closeVariables
        
    def uniqueDict(self,tup):
        #returns a dictionary with unique values from tuples and frequency of each value
        result={}
        for x in tup:
            if(x in result):
                result[x]+=1
            else:
                result[x]=1
        return result
    
    def dataAnalyzer(self):
        #the core funtion which conducts PCA on the variables and calculate the principal components
        #the attributes are then projected to primary principal components to identify relation among the attributes
        #the related attributes are used to identify the relations among them.
        
        inputData=self.readInput()
        
        self.processGraph()
        
        #print(inputData)
        
        for x in inputData:
#          #  print(savgol_filter(inputData[x],5,2))   
            inputData[x]=savgol_filter(inputData[x],5,2)
#           # print(x)
            
#        print(inputData)
        
        #print(inputData)
        #recieve the projection of attributes in the principal components
        PCAdata=self.analyzer(inputData)
        
        #calculates the distance between individual attribues in the principal component space
      
        distanceMatrix={}
        count=0
        distanceSum=0
        
        for a in range(len(PCAdata["0"])):
            for b in range(a,len(PCAdata["0"])):
                #print(PCAdata.index[a],PCAdata.index[b])
                if(PCAdata.index[a]!=PCAdata.index[b]):
                    count+=1
                    distance=self.pointsDistance(PCAdata["0"][a],PCAdata["1"][a],PCAdata["0"][b],PCAdata["1"][b])
                    distanceSum+=distance
                    distanceMatrix[PCAdata.index[a]+"&&"+PCAdata.index[b]]=distance
        
        
        averageDistance=distanceSum/count
        
#        print(averageDistance)
 #       print(distanceMatrix)
        
        
        #identifies the closely related attributes(lesser distance between them in principal component space)
        closeVariables=self.calculateCloselyCorrealtedVariables(averageDistance,distanceMatrix)
        
        print(closeVariables)
    
        #degrees is used to specify the different degrees of equation that need to be checked 
        degrees=[2]
        
        
        #the function that finds the relation between the attributes by analyzing related variables
        self.findRelation(degrees,closeVariables,inputData)
        
        if(os.path.exists(os.path.join(os.getcwd(),"Logs"))==False):
            os.mkdir("Logs")
        
        now =datetime.datetime.now()
        
        filename=str(now.month)+"_"+str(now.day)+"_"+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"_.txt"
        with open(os.path.join("Logs",filename),'w') as fn:
            fn.write(str(now.month)+"/"+str(now.day)+"/"+str(now.year)+" "+str(now.hour)+":"+str(now.minute)+":"+str(now.second))
            fn.write('\n-----------------------------------------------\n')
            fn.write("Data Size:"+str(len(inputData)))
            fn.write("Degrees tested: "+str(degrees))
            fn.write("\n-----------------------------------------------\n")
            fn.write("Possible relations among the attributes are:\n")
            for formula in self.formulas:
                fn.write(formula)
                fn.write("\n")
            fn.write("----------------------------------------------\n")
            fn.close()
        
        
            
        print("Possible relations among the attributes are:")
        
        #result of the analysis
        for formula in self.formulas:
            print(formula)
            
        #print(list(set(self.formulas))) 
                    #square root get combination of tuple and root the values of combination
            
    def combinationDistance(self,attribute1,attribute2):
        if(attribute1 not in self.vertexMap or attribute2 not in self.vertexMap):
            return 10000
        i=self.vertexMap[attribute1]
        j=self.vertexMap[attribute2]
        
        return self.distanceBetweenSensors[i][j]
        
    def findRelation(self,degrees,closeVariables,inputData):
        #the function finds relation between variables by checking combination of variables for degrees provided
        
        
        self.formulas=[]
        print(closeVariables.keys())
        #iterate through different degrees
        for degree in degrees:
            print(degree)
            for result in closeVariables.keys():
#                if(result!='q4'):
#                    continue
                print(result)
                #generate combinations of variables related to every variable
                combinations=itertools.combinations_with_replacement(closeVariables[result],degree)
               # combinationLength=len(combinations)
                counter=1
                for combination in combinations:
#                    if('q3' not in combination or 'KIi' not in combination):
#                        continue
                    #get dictionary of type {unique_variable:frequency}
                   #print(combination)
                    # print(counter,"/",combinationLength," processed for ",result," in ",degree)
                   runFlag=0
                   for x in combination:
                        if(self.combinationDistance(x,result)>4):
                            runFlag=1
                            break
                    
                   if(runFlag):
                        continue
                    
                   uniqueVariableCount=self.uniqueDict(combination)
                   uniqueVariableCount[result]=1
                    #temp=[x for x in tupCount.keys()]
                      
                    #generate temporary dataframe of the the variables under check
                   tempDf=inputData[[x for x in uniqueVariableCount.keys()]]    
                   flag=0
                    #iterate through the variables and generate the product of data for respective attributes
                   for var in uniqueVariableCount.keys():
                       if(var!=result):
                           for freq in range(uniqueVariableCount[var]):
                                if(flag==0):
                                    tempDf['convertedOutput']=tempDf[var]
                                    flag=1
                                else:
                                    tempDf['convertedOutput']*=tempDf[var]
                    
                    
                    #find the average percent of difference between target variable and result of combination of related variables to target 
                   diff=0
                   for index,row in tempDf.iterrows():
                        diff+=abs(row[result]-row['convertedOutput'])/abs(row[result])                    
                    #check if the differnce between variables is less than 25 percent
#                    print(tempDf)
                   if(diff/len(tempDf)<=0.25):
                        y=""
                        for x in combination:
                            y=y+x+"*"
                        y=y[0:len(y)-1]
                        #print(y+"="+result,diff/len(tempDf))
                        self.formulas.append(y+"="+result)
                   else:
                        
                        #checking if target has relation with square root of individual variables
                        for x in range(1,degree):
                            rootVariables=list(itertools.combinations_with_replacement(uniqueVariableCount,x))
                            for rootVariable in rootVariables:
                                rootVariableList=list(rootVariable)
                                if(result in rootVariableList):
                                    continue
                                flag=0
                                for Variable in uniqueVariableCount:
                                    if(Variable in rootVariableList):
                                        rootVariableList.remove(Variable)
                                        if(flag==0):
                                            tempDf['convertedOutputRoot']=np.sqrt(tempDf[Variable])
                                            flag=1
                                        else:
                                            tempDf['convertedOutputRoot']*=np.sqrt(tempDf[Variable])
                                    else:
                                        if(flag==0):
                                            tempDf['convertedOutputRoot']=tempDf[Variable]
                                            flag=1
                                        else:
                                            tempDf['convertedOutputRoot']*=tempDf[Variable]
                        
                        
                                diff=0
                                for index,row in tempDf.iterrows():
                                    diff+=abs(row[result]-row['convertedOutputRoot'])/row[result]
                                
                                
                                
                                if(diff/len(tempDf)<=0.25):
                                    y=""
                                    rootVariableList=list(rootVariable)
                                    for x in uniqueVariableCount:
                                        if(x in rootVariableList):
                                            rootVariableList.remove(x)
                                            y=y+x+"^1/2"+"*"
                                        else:
                                            y=y+x+"*"
                                
                                    y=y[0:len(y)-1]
                                    self.formulas.append(y+"="+result)
                        
                        for x in range(2,degree):
                            rootVariables=list(itertools.combinations_with_replacement(uniqueVariableCount,x))
                            for rootVariable in rootVariables:
                                rootVariableList=list(rootVariable)
                                if(result in rootVariableList):
                                    continue
                                flag=0
                                for rv in rootVariableList:
                                    if(flag==0):
                                            tempDf['convertedOutputRootGroup']=tempDf[rv]
                                            flag=1
                                    else:
                                            tempDf['convertedOutputRootGroup']*=tempDf[rv]
                                
                                
                                tempDf['convertedOutputRootGroup']=np.sqrt(tempDf['convertedOutputRootGroup'])
                                
                                for Variable in uniqueVariableCount:
                                    if(Variable in rootVariableList):
                                        rootVariableList.remove(Variable)
                                    else:
                                        tempDf['convertedOutputRootGroup']*=tempDf[Variable]
                       
                                diff=0
                                for index,row in tempDf.iterrows():
                                    diff+=abs(row[result]-row['convertedOutputRootGroup'])/row[result]
                                
                                
                                if(diff/len(tempDf)<=0.25):
                                    y="("
                                    rootVariableList=list(rootVariable)
                                    for rv in rootVariableList:
                                        y=y+rv+"*"
                                    y=y[0:len(y)-1]
                                    y=y+")^1/2*"
                                    for x in uniqueVariableCount:
                                        if(x in rootVariableList):
                                            rootVariableList.remove(x)
                                        else:
                                            y=y+x+"*"
                                            
                                    y=y[0:len(y)-1]
                                    self.formulas.append(y+"="+result)
        
        self.formulas=list(set(self.formulas))
        
        
    def modeProjection(self,data):
        norm=0
        for index in range(len(data["0"])):
            x=data["0"][index]
            y=data["1"][index]
            norm+=math.sqrt(x*x+y*y)  
        return float(norm)/len(data["0"])    
        
    
    def analyzer(self,inputData):  
              
        # Normalization of data
        dataNorm = (inputData - inputData.mean()) / inputData.std()
        print(type(dataNorm['Kinj'][0]))
        # PCA
        pca = PCA(n_components=10)
        pca.fit_transform(dataNorm.values)
        
        pcaTranspose = np.transpose(pca.components_)
        cols = [str(x) for x in range(len(panda.Series(pca.explained_variance_ratio_)))]
        
        relationMatrix = panda.DataFrame(pcaTranspose, columns=cols, index=dataNorm.columns)
#        
        return relationMatrix
#

if __name__=='__main__':
    try:
        filepath=sys.argv[1]
        distanceFile=sys.argv[2]
    except:
        print("")
        print("Usage: analyzer_comb.py <filepath>")
        exit() 
    
    analyzerObject=dataAna(filepath,distanceFile)
    analyzerObject.dataAnalyzer()
    