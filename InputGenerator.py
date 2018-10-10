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
            
        
                        
            
#            fprgoutTemp=self.lowerBound
#            _96hqlTemp=self.lowerBound
#            fsroutTemp=self.lowerBound
#            KinjTemp=self.lowerBound
#            cpaTemp=self.lowerBound
#            KITemp=self.lowerBound
#            KIiTemp=self.lowerBound
#            KldashTemp=self.lowerBound
#            p3Temp=self.lowerBound
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
            #while(KldashCount!=self.dataSize or cpaCount!=self.dataSize or p3Count!=self.dataSize or fprgoutCount!=self.dataSize or _96hqlCount!=self.dataSize or fsroutCount!=self.dataSize or KinjCount!=self.dataSize or KICount!=self.dataSize or KIiCount!=self.dataSize):
#                fprgout=random.uniform(self.lowerBound,self.upperBound)
#                fpg2=self.f(fprgout)
#                p3=random.uniform(p3Temp,fpg2)
#                cpa=random.uniform(cpaTemp,p3)
#                Kldash=random.uniform(KldashTemp,p3)
#                _96hql=random.uniform(_96hqlTemp,self.upperBound)
#                fsrout=random.uniform(fsroutTemp,self.upperBound)
#                Kinj=random.uniform(KinjTemp,self.upperBound)
#                KI=random.uniform(KITemp,self.upperBound)
#                KIi=random.uniform(KIiTemp,self.upperBound)
#                fprgoutCount+=1
#                _96hqlCount+=1
#                fsroutCount+=1
#                KinjCount+=1
#                KICount+=1
#                KIiCount+=1
                
             #   if(random.randint(0,1) and cpaCount!=self.dataSize):
                 cpa=cpaTemp[cpaCount]
                 cpaCount+=1
                #else:
                 #   cpa=""
                #if(random.randint(0,1) and p3Count!=self.dataSize):
                 p3=p3Temp[p3Count]
                 p3Count+=1
                #else:
                 #   p3=""
                #if(random.randint(0,1) and KldashCount!=self.dataSize):
                 Kldash=KldashTemp[KldashCount]
                 KldashCount+=1
                #else:
                 #   Kldash=""
                #if(random.randint(0,1) and fprgoutCount!=self.dataSize):
                 fprgout=fprgoutTemp[fprgoutCount]
                    
#                    if(fprgoutCount<self.dataSize/2):
#                        fprgout=random.uniform(fprgoutTemp,self.upperBound)
#                        fprgoutTemp=fprgout
#                        fpg2=self.f(fprgout)#R11
#                        p3=random.uniform(p3Temp,fpg2)
#                        p3Temp=p3
#                        cpa=random.uniform(cpaTemp,p3)
#                        cpaTemp=cpa
#                        Kldash=random.uniform(KldashTemp,p3)
#                        KldashTemp=Kldash
#                    else:
#                        fprgout=random.uniform(self.lowerBound,fprgoutTemp)
#                        fpg2=self.f(fprgout)#R11
#                        p3=random.uniform(self.lowerBound,fpg2)
#                        cpa=random.uniform(self.lowerBound,p3Temp)
#                        Kldash=random.uniform(self.lowerBound,p3Temp)
#                        fprgoutTemp=fprgout
#                        p3Temp=p3
#                        cpaTemp=cpa
#                        KldashTemp=Kldash
                 fprgoutCount+=1
                #else:
                 #   fprgout=""
                #if(random.randint(0,1) and _96hqlCount!=self.dataSize):
                 _96hql=_96hqlTemp[_96hqlCount]
#                    if(_96hqlCount<self.dataSize/2):
#                        _96hql=random.uniform(_96hqlTemp,self.upperBound)
#                        _96hqlTemp=_96hql
#                    else:
#                        _96hql=random.uniform(self.lowerBound,_96hqlTemp)
#                        _96hqlTemp=_96hql
                 _96hqlCount+=1
                #else:
                 #   _96hql=""
                #if(random.randint(0,1) and fsroutCount!=self.dataSize):
                 fsrout=fsroutTemp[fsroutCount]
#                    if(fsroutCount<self.dataSize/2):
#                        fsrout=random.uniform(fsroutTemp,self.upperBound)
#                        fsroutTemp=fsrout
#                    else:
#                        fsrout=random.uniform(self.lowerBound,fsroutTemp)
#                        fsroutTemp=fsrout
                 fsroutCount+=1
                #else:
                 #   fsrout=""
                #if(random.randint(0,1) and KinjCount!=self.dataSize):
                 Kinj=KinjTemp[KinjCount]
#                    if(KinjCount<self.dataSize/2):
#                        Kinj=random.uniform(KinjTemp,self.upperBound)
#                        KinjTemp=Kinj
#                    else:
#                        Kinj=random.uniform(self.lowerBound,KinjTemp)
#                        KinjTemp=Kinj
                 KinjCount+=1
                #else:
                 #   Kinj=""
                #if(random.randint(0,1) and KICount!=self.dataSize):
                 KI=KITemp[KICount]
#                    if(KICount<self.dataSize/2):
#                        KI=random.uniform(KITemp,self.upperBound)
#                        KITemp=KI
#                    else:
#                        KI=random.uniform(self.lowerBound,KITemp)
#                        KITemp=KI
                 KICount+=1
                #else:
                 #   KI=""
                #if(random.randint(0,1) and KIiCount!=self.dataSize):
                 KIi=KIiTemp[KIiCount]
#                    if(KIiCount<self.dataSize/2):
#                        KIi=random.uniform(KIiTemp,self.upperBound)
#                        KIiTemp=KIi
#                    else:
#                        KIi=random.uniform(self.lowerBound,KIiTemp)
#                        KIiTemp=KIi
                 KIiCount+=1
#                else:
#                    KIi=""
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
