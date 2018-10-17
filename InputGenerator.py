# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:29:01 2018

@author: amal
"""

import numpy as np
import math
import random
import sys
import csv


class dataGen:
    
    def __init__(self,lowerBound,upperBound,dataSize):
        self.lowerBound=float(lowerBound)
        self.upperBound=float(upperBound)
        self.dataSize=int(dataSize)
    
    def f(self,input1,input2=1,input3=1):
        return math.sqrt(input1*input2*input3)
    
    def f_inverse(self,output,input1,input2=1):
        return output*output/(input1*input2)
    
    def sortBitonic(self,arr,mean):
        y=[arr[arr<mean],arr[arr>mean]]
        y[0].sort()
        y[1].sort()
        z1=[m for m in y[0]]
        z2=[m for m in y[1]]
        z2.reverse()
        z1.extend(z2)
        return z1
        
    
    def dataGenerator(self):
        with open('data1.csv','wb') as csvf:
            fields=['fprgout','_96hql','fsrout','Kinj','cpa','KI','KIi','Kldash','p3']
            writer=csv.DictWriter(csvf,fieldnames=fields)
            writer.writeheader()
            data={}
            mean=(self.lowerBound+self.upperBound)/2
            fprgoutTemp=np.random.normal(mean,math.sqrt(mean),self.dataSize)
            _96hqlTemp=np.random.normal(mean,math.sqrt(mean),self.dataSize)
            fsroutTemp=np.random.normal(mean,math.sqrt(mean),self.dataSize)
            KinjTemp=np.random.normal(mean,math.sqrt(mean),self.dataSize)
            cpaTemp=np.random.normal(mean,math.sqrt(mean),self.dataSize)
            KITemp=np.random.normal(mean,math.sqrt(mean),self.dataSize)
            KIiTemp=np.random.normal(mean,math.sqrt(mean),self.dataSize)
            KldashTemp=np.random.normal(mean,math.sqrt(mean),self.dataSize)
            p3Temp=np.random.normal(mean,math.sqrt(mean),self.dataSize)
            
            fprgoutTemp=self.sortBitonic(fprgoutTemp,mean)
            _96hqlTemp=self.sortBitonic(_96hqlTemp,mean)
            fsroutTemp=self.sortBitonic(fsroutTemp,mean)
            KinjTemp=self.sortBitonic(KinjTemp,mean)
            cpaTemp=self.sortBitonic(cpaTemp,mean)
            KIiTemp=self.sortBitonic(KIiTemp,mean)
            KITemp=self.sortBitonic(KITemp,mean)
            KldashTemp=self.sortBitonic(KldashTemp,mean)
            p3Temp=self.sortBitonic(p3Temp,mean)

            fprgoutCount=0
            _96hqlCount=0
            fsroutCount=0
            KinjCount=0
            KICount=0
            KIiCount=0
            p3Count=0
            cpaCount=0
            KldashCount=0
            for i in range(0,self.dataSize):

                 cpa=cpaTemp[cpaCount]
                 cpaCount+=1
                 p3=p3Temp[p3Count]
                 p3Count+=1
                 Kldash=KldashTemp[KldashCount]
                 KldashCount+=1
                 fprgout=fprgoutTemp[fprgoutCount]
                 fprgoutCount+=1
                 _96hql=_96hqlTemp[_96hqlCount]
                 _96hqlCount+=1
                 fsrout=fsroutTemp[fsroutCount]
                 fsroutCount+=1
                 Kinj=KinjTemp[KinjCount]
                 KinjCount+=1
                 KI=KITemp[KICount]
                 KICount+=1
                 KIi=KIiTemp[KIiCount]
                 KIiCount+=1
                 data['fprgout']=fprgout
                 data['_96hql']=_96hql
                 data['fsrout']=fsrout
                 data['Kinj']=Kinj
                 data['cpa']=cpa
                 data['KI']=KI
                 data['KIi']=KIi
                 data['Kldash']=Kldash
                 data['p3']=p3
                 writer.writerow(data)
            
if __name__=='__main__':
    try:
        lowerBound=sys.argv[1]
        upperBound=sys.argv[2]
        dataSize=sys.argv[3]
    except:
        print ""
        print "Usage: datagenerator.py <lowerBound> <upperBound> <dataSize>"
        exit() 
    
    dataGenObject=dataGen(lowerBound,upperBound,dataSize)
    dataGenObject.dataGenerator()
