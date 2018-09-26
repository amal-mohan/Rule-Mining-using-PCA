# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 14:23:50 2018

@author: amal
"""

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
    
    def dataGenerator(self):
        with open('data.csv','wb') as csvf:
            fields=['fpg2','p3','fqg','q2','q3','q4','fsg','fsgr','fag','fagr']
            writer=csv.DictWriter(csvf,fieldnames=fields)
            writer.writeheader()
            data={}
            for i in range(0,self.dataSize):
                fprgout=random.uniform(self.lowerBound,self.upperBound)
                _96hql=random.uniform(self.lowerBound,self.upperBound)
                fsrout=random.uniform(self.lowerBound,self.upperBound)
                fpg2=self.f(fprgout)#R11
                fsgr=self.f(fprgout,fpg2,_96hql)#R10
                fagr=self.f_inverse(fsgr,_96hql)#R8
                fsg=self.f(fsrout,_96hql)#R9
                fag=self.f_inverse(fsg,_96hql)#R7
                Kinj=random.uniform(self.lowerBound,self.upperBound)
                p3=random.uniform(self.lowerBound,fpg2)
                cpa=random.uniform(self.lowerBound,p3)
                q3=Kinj*math.sqrt(p3-cpa)#R1
                KI=random.uniform(self.lowerBound,self.upperBound)
                q2=q3/KI#R4
                q1=fsg*math.sqrt(fpg2-p3)#R3
                KIi=random.uniform(self.lowerBound,self.upperBound)
                q4=q3*KIi#R2
                Kldash=random.uniform(self.lowerBound,p3)
                fqg=q1/Kldash#R6
                data['fpg2']=fpg2
                data['p3']=p3
                data['fqg']=fqg
                data['q2']=q2
                data['q3']=q3
                data['q4']=q4
                data['fsg']=fsg
                data['fsgr']=fsgr
                data['fag']=fag
                data['fagr']=fagr
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
