
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 11:54:40 2018

@author: amal
"""
 
#time of pca and full process--adjustment
#accuracy think
#performance improvement
#get more data for more testing

#steps for input
# missing values of input

import datetime
import json
import math
import os
import sys
import pandas as panda
from sklearn.decomposition import PCA
import numpy as np
import itertools
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import savgol_filter
import multiprocessing
from multiprocessing import Pool
from contextlib import closing
from contextlib import contextmanager

import copy_reg
import types

def threadResolver(method):
    if method.im_self is None:
        return getattr, (method.im_class, method.im_func.func_name)
    else:
        return getattr, (method.im_self, method.im_func.func_name)

copy_reg.pickle(types.MethodType, threadResolver)

#implemented only for multiplication, generalization requires changes
class expression:
    
    def __init__(self,listOfAttributes,result,exp):
        self.listofAttributes=listOfAttributes
        self.result=result
        self.exp=exp
        self.attributeSet={}
        self.sortAttributes()
    
    def sortAttributes(self):
        for attribute in self.listofAttributes:
            if(attribute in self.attributeSet):
                self.attributeSet[attribute]+=1
            else:
                self.attributeSet[attribute]=1
       
        

class dataAna:

    def __init__(self,configurationFilePath):
        self.recall=0
        with open(configurationFilePath,"r")  as jsonFile:
            configurationFile=json.load(jsonFile)
            self.filepath=configurationFile['input-file-path']
            self.graphFilePath=configurationFile['graph-file-location']
            self.masterformulasLocation=configurationFile['orginal-formulas-list']
            self.approach=configurationFile['approach']
            if 'graph-use' in configurationFile:
                self.graphuse=configurationFile['graph-use']
            else:
                self.graphuse=1
            if 'max-threads' in configurationFile:
                self.maxThreads=configurationFile['max-threads']
            else:
                self.maxThreads=10
            if 'max-distance-between-sensors' in configurationFile:
                self.maxDistance=configurationFile['max-distance-between-sensors']
            else:
                self.maxDistance=3
            if 'degrees' in configurationFile:
                self.degrees=configurationFile['degrees']
            else:
                self.degrees=[2]
        self.distanceBetweenSensors=[]
        
    def readInput(self):
        #reads input data
        #interpolates the missing values and drop not available values
        inputData=panda.read_csv(self.filepath)
        for x in inputData:
            inputData=inputData.interpolate(method='linear')
        inputData.dropna(inplace=True)
        inputData.reset_index(drop=True,inplace=True)
        inputData=inputData.drop("time",axis=1)
        return inputData
        
    def pointsDistance(self,x1,y1,x2,y2):
        # calculates the eucledian distance between input distance
        a=abs(math.degrees(math.atan(y2/x2)))
        b=abs(math.degrees(math.atan(y1/x1)))
#        x=abs(x1-x2)
#        y=abs(y1-y2)
       # print(a,b)
        return abs(a-b)
#        return x+y
    
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
            if(distanceMatrix[x]<=10 or distanceMatrix[x]>=350):
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

    def getCloselyCorreleatedVariables(self,PCAdata):
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
        
        
        
        #identifies the closely related attributes(lesser distance between them in principal component space)
        closeVariables=self.calculateCloselyCorrealtedVariables(averageDistance,distanceMatrix)
        
  #      print(closeVariables)
        
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
    
    
    def approachA(self):
        self.PCAstarttime=datetime.datetime.now()
        PCAdata=self.analyzer(self.inputData)
        self.PCAendtime=datetime.datetime.now()
 

    
        #degrees is used to specify the different degrees of equation that need to be checked 
        
        closeVariables=self.getCloselyCorreleatedVariables(PCAdata)
        
        #the function that finds the relation between the attributes by analyzing related variables
        self.findRelation(closeVariables)

    
    def approachB(self):
        self.PCAendtime=datetime.datetime.now()
        self.PCAstarttime=datetime.datetime.now()
        self.formulas=[]
        comb=[]
        fields=self.inputData.keys()
        for degree in self.degrees:
            combinations=itertools.combinations_with_replacement(fields,degree)
            
            for combination in combinations:
                name=""
                for x in combination:               
                    name=name+x+"*"
                
                name=name[0:len(name)-1]

                comb.append(name)
                flag=0
                for x in combination:
                    if(flag==0):
                        self.inputData[name]=self.inputData[x]
                    else:
                        self.inputData[name]*=self.inputData[x] 
                        
        
#        print(self.inputData)
        
        PCAdata=self.analyzer(self.inputData)
        
        for a in range(len(PCAdata["0"])):
            print(PCAdata["0"][a],PCAdata["1"][a])
#            print(PCAdata["0"],PCAdata["1"])
        
        closeVariables=self.getCloselyCorreleatedVariables(PCAdata)
        form=[]
        i=0

        for x in comb:
            if(x in closeVariables):
    #            print(x)
                #print(closeVariables[x])
                xRelated=closeVariables[x]
                for y in xRelated:
                    if(y not in comb  and y not in x.split("*")):
                        
                        diff=0
 #                       print(i)
                        i=i+1
#                        for index,row in self.inputData.iterrows():
#                            diff+=abs(row[y]-row[x])/abs(row[y])                    
#                        
#                        if(diff/len(self.inputData)<=0.25):
                        form.append(x+"="+y)    
#        print(comb)
#        print(form)
                    
               
        
        
    def dataAnalyzer(self):
        #the core funtion which conducts PCA on the variables and calculate the principal components
        #the attributes are then projected to primary principal components to identify relation among the attributes
        #the related attributes are used to identify the relations among them.
        
        starttime=datetime.datetime.now()
        
        self.inputData=self.readInput()
        
        self.processGraph()
        
        #print(inputData)
        
        for x in self.inputData:
            self.inputData[x]=savgol_filter(self.inputData[x],5,2)
        #recieve the projection of attributes in the principal components
        
        self.approachA()
        
        if(os.path.exists(os.path.join(os.getcwd(),"Logs"))==False):
            os.mkdir("Logs")
        
        now =datetime.datetime.now()
        
        diff=now-starttime
        
        diffmins=diff.total_seconds()/60
        
        PCAdatatime=self.PCAendtime-self.PCAstarttime
        
        diffPCA=PCAdatatime.total_seconds()/60
        
 #       self.caculateAccuracy()
        
        filename=str(now.month)+"_"+str(now.day)+"_"+str(now.year)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+str(now.second)+"_.txt"
        with open(os.path.join("Logs",filename),'w') as fn:
            fn.write(str(now.month)+"/"+str(now.day)+"/"+str(now.year)+" "+str(now.hour)+":"+str(now.minute)+":"+str(now.second))
            fn.write('\n-----------------------------------------------\n')
            fn.write("Data Size:"+str(len(self.inputData)))
            fn.write("\n-----------------------------------------------\n")
            fn.write("Total PCA Time: "+str(diffPCA))
            fn.write("\n-----------------------------------------------\n")
            fn.write("Total Time taken: "+str(diffmins))
            fn.write("\n-----------------------------------------------\n")
            fn.write("Degrees tested: "+str(self.degrees))
            fn.write("\n-----------------------------------------------\n")
            fn.write("Possible relations among the attributes are:\n")
            for formula in self.formulas:
                fn.write(formula)
                fn.write("\n")
            fn.write("----------------------------------------------\n")
            fn.write("Recall:"+str(self.recall))
            fn.close()
        
        
            
        print("Possible relations among the attributes are:")
        
        #result of the analysis
        for formula in self.formulas:
            print(formula)
            
        #print(list(set(self.formulas))) 
                    #square root get combination of tuple and root the values of combination
        
    def caculateAccuracy(self):
        masterFormulas=self.masterFormulaList()
        inter=self.intersection(self.formulas,masterFormulas)
        if(len(inter)==0):
            self.recall=0
        else:
            self.recall=len(inter)/len(masterFormulas)
    
    def intersection(self,list1,list2):
        temp=set(list2)
        res=[i for i in list1 if i in temp]
        return res
    
    def masterFormulaList(self):
        
        return
    
    def combinationDistance(self,attribute1,attribute2):
        if(attribute1 not in self.vertexMap or attribute2 not in self.vertexMap):
            return 10000
        i=self.vertexMap[attribute1]
        j=self.vertexMap[attribute2]
        
        return self.distanceBetweenSensors[i][j]
        
    def relationThread(self,combination,result,degree):
        uniqueVariableCount=self.uniqueDict(combination)
        uniqueVariableCount[result]=1
          
        #generate temporary dataframe of the the variables under check
        tempDf=self.inputData[[x for x in uniqueVariableCount.keys()]]    
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
        
        
            
        diff=0
        #find the average percent of difference between target variable and result of combination of related variables to target 
        for index,row in tempDf.iterrows():
            diff+=abs(row[result]-row['convertedOutput'])/abs(row[result])                    
        #check if the differnce between variables is less than 25 percent
#                    print(tempDf)
#        attributeList=[]

#        if(df[result].corr(df['convertedOutput'])>0.90):
#            corre=df[result].corr(df['convertedOutput'])
        if(diff/len(tempDf)<=0.25):
            y=""
            for x in combination:
                y=y+x+"*"
#                attributeList.append((x,1))
            y=y[0:len(y)-1]
            return (y+"="+str(tempDf[result].corr(tempDf['convertedOutput']))+result)
        #expression(attributeList,result,"*") #(y+"="+result)
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
                    
                    
#                    if(df[result].corr(df['converterdOutput'])>0.90):
              
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
                        return(y+"="+str(tempDf[result].corr(tempDf['convertedOutputRoot']))+result)
                        
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
                   
#                    if(df[result].corr(df['converterdOutput'])>0.90):
     
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
                        return (y+"="+str(tempDf[result].corr(tempDf['convertedOutputRootGroup']))+result)
        return ""
        
    def threadManager(self,args):
        return self.relationThread(*args)

    @contextmanager
    def poolcontext(*args, **kwargs):
        pool = multiprocessing.Pool(*args, **kwargs)
        yield pool
        pool.terminate()

    def findRelation(self,closeVariables):
        #the function finds relation between variables by checking combination of variables for degrees provided
        self.formulas=[]
        print(closeVariables.keys())
        #iterate through different degrees
        for degree in self.degrees:
            print(degree)
            for result in closeVariables.keys():
                print(result)
                #generate combinations of variables related to every variable
                combinations=itertools.combinations_with_replacement(closeVariables[result],degree)
               # combinationLength=len(combinations)
                counter=1
                cl=[]
                for combination in combinations:
                   runFlag=0
                   for x in combination:
                        if(self.combinationDistance(x,result)>self.maxDistance and self.graphuse==1):
                            runFlag=1
                            break
                    
                   if(runFlag):
                        continue
                    
                   if(counter!=self.maxThreads):
                       parameterList=(combination,result,degree)
                       cl.append(parameterList)
                       counter+=1
                   else:
                       with closing(Pool(self.maxThreads)) as pool:	
                           formulas=pool.map(self.threadManager, cl)
                       cl=[]
                       counter=0
                       for formula in formulas:
                           if(formula)!="":
                               self.formulas.append(formula)
                if(cl):
                    with closing(Pool(len(cl))) as pool:	
                           formulas=pool.map(self.threadManager, cl)
                    cl=[]
                    counter=0
                    for formula in formulas:
                        if(formula)!="":
                            self.formulas.append(formula)
                      
                           
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
        # PCA
#        print dataNorm
        pca = PCA(n_components=10)
        pca.fit_transform(dataNorm.values)
        
        pcaTranspose = np.transpose(pca.components_)
        cols = [str(x) for x in range(len(panda.Series(pca.explained_variance_ratio_)))]
        
        relationMatrix = panda.DataFrame(pcaTranspose, columns=cols, index=dataNorm.columns)
#        print(relationMatrix)
        return relationMatrix


if __name__=='__main__':
    try:
        configurationFile=sys.argv[1]
    except:
        print("")
        print("Usage: analyzer_comb.py <configurationFilePath>")
        exit() 
    
    analyzerObject=dataAna(configurationFile)
    analyzerObject.dataAnalyzer()
    