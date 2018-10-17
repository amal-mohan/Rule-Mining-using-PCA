# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 13:37:01 2018

@author: amal
"""

#todo
#timeseries data
#noise propogation

import math
import sys
import csv
import pandas as panda
import random
import numpy as np
from sklearn.linear_model import LinearRegression

class dataGen:

    def __init__(self,filepath,noisePropogation):
        self.filepath=filepath
        self.noisePropogation=noisePropogation
    
    def noiseAdder(self,value):
        result=value
        if(random.randint(0,1)):
            rangeValue=value*0.10
            if(rangeValue>5):
                rangeValue=5
            noise=random.uniform(0,rangeValue)
            if(random.randint(0,1)):
                result+=noise
            else:
                result-=noise   
        return result
            
    def f(self,input1,input2=1,input3=1):
        return math.sqrt(abs(input1*input2*input3))
    
    def f_inverse(self,output,input1,input2=1):
        return output*output/(input1*input2)
    
    def dataGenerator(self):
        
        inputData=panda.read_csv(self.filepath)
        inputData=inputData.interpolate()
        inputData.dropna(inplace=True)
        inputData.reset_index(drop=True,inplace=True)
        dataLength=len(inputData)
        

        with open('data.csv','wb') as csvf:
            fields=['fpg2','p3','fqg','q1','q2','q3','q4','fsg','fsgr','fag','fagr','fprgout','_96hql','fsrout','Kinj','cpa','KI','KIi','Kldash']
            writer=csv.DictWriter(csvf,fieldnames=fields)
            writer.writeheader()
            data={}
            s=[]
            row=len(inputData)
            col=len(fields)
                                


            noise_fpg2=np.random.normal(0,0.1,(row))
            noise_p3=np.random.normal(0,0.1,(row))
            noise_fqg=np.random.normal(0,0.1,(row))
            noise_q1=np.random.normal(0,0.1,(row))
            noise_q2=np.random.normal(0,0.1,(row))
            noise_q3=np.random.normal(0,0.1,(row))
            noise_q4=np.random.normal(0,0.1,(row))
            noise_fsg=np.random.normal(0,0.1,(row))
            noise_fsgr=np.random.normal(0,0.1,(row))
            noise_fag=np.random.normal(0,0.1,(row))
            noise_fagr=np.random.normal(0,0.1,(row))
            noise_fprgout=np.random.normal(0,0.1,(row))
            noise__96hql=np.random.normal(0,0.1,(row))
            noise_fsrout=np.random.normal(0,0.1,(row))
            noise_Kinj=np.random.normal(0,0.1,(row))
            noise_cpa=np.random.normal(0,0.1,(row))
            noise_KI=np.random.normal(0,0.1,(row))
            noise_KIi=np.random.normal(0,0.1,(row))
            noise_Kldash=np.random.normal(0,0.1,(row))
        


            for i in range(0,dataLength):
                #generating data
                
                
                fprgout=inputData.fprgout.loc[i]
                _96hql=inputData._96hql.loc[i]
                fsrout=inputData.fsrout.loc[i]
                Kinj=inputData.Kinj.loc[i]
                p3=inputData.p3.loc[i]
                cpa=inputData.cpa.loc[i]
                KI=inputData.KI.loc[i]
                KIi=inputData.KIi.loc[i]
                Kldash=inputData.Kldash.loc[i]
                
                                
                fpg2=self.f(fprgout)#R11
                fsgr=self.f(fprgout,fpg2,_96hql)#R10
                fagr=self.f_inverse(fsgr,_96hql)#R8
                fsg=self.f(fsrout,_96hql)#R9
                fag=self.f_inverse(fsg,_96hql)#R7
                q3=Kinj*math.sqrt(abs(p3-cpa))#R1
                q2=q3/KI#R4
                q1=fsg*math.sqrt(abs(fpg2-p3))#R3
                q4=q3*KIi#R2
                fqg=q1/Kldash#R6                
                
            
                #storing data
                data['fpg2']=fpg2+noise_fpg2[i]
                data['p3']=p3+noise_p3[i]
                data['fqg']=fqg+noise_fqg[i]
                data['q2']=q2+noise_q2[i]
                data['q3']=q3+noise_q3[i]
                data['q1']=q1+noise_q1[i]                
                data['q4']=q4+noise_q4[i]
                data['fsg']=fsg+noise_fsg[i]
                data['fsgr']=fsgr+noise_fsgr[i]
                data['fag']=fag+noise_fag[i]
                data['fagr']=fagr+noise_fagr[i]
                data['fprgout']=fprgout+noise_fprgout[i]
                data['_96hql']=_96hql+noise__96hql[i]
                data['fsrout']=fsrout+noise_fsrout[i]
                data['Kinj']=Kinj+noise_Kinj[i]
                data['cpa']=cpa+noise_cpa[i]
                data['KI']=KI+noise_KI[i]
                data['KIi']=KIi+noise_KIi[i]
                data['Kldash']=Kldash+noise_Kldash[i]
                
                writer.writerow(data)

            csvf.close()
            
            
            df=panda.read_csv("data.csv")
            a=[x*5 for x in range(1,len(df)+1)]
            df = df.assign(time=panda.Series(a).values)
            timeSeriesGen={}
            alltimeSet=set([])
            for l in df:
                if(l=="time"):
                    continue
                timeSeriesGen[l]={}
                s=np.random.randint(1,a[len(a)-1],len(a))
                alltimeSet=alltimeSet.union(set(s))
                reg=LinearRegression().fit(np.array([[x] for x in a]),df[l])
                for ele in s:
                    timeSeriesGen[l][ele]=reg.predict(np.array([[ele]]))[0]
            
            
        with open('data.csv','wb') as csvf:
            fields=['time','fpg2','p3','fqg','q1','q2','q3','q4','fsg','fsgr','fag','fagr','fprgout','_96hql','fsrout','Kinj','cpa','KI','KIi','Kldash']
            writer=csv.DictWriter(csvf,fieldnames=fields)
            writer.writeheader()
            data={}
            alltimeSet=list(alltimeSet)
            alltimeSet.sort()
            for secs in alltimeSet:
                data['time']=secs
                for attribute in timeSeriesGen:
                    if(secs in timeSeriesGen[attribute]):
                        data[attribute]=timeSeriesGen[attribute][secs]
                    else:
                        data[attribute]=""
                writer.writerow(data)
            
            csvf.close()
                    
        
if __name__=='__main__':
    try:
        filepath=sys.argv[1]
        noisePropogation=sys.argv[2]
    except:
        print ""
        print "Usage: datagenerator.py <filepath> <noisepropogation>"
        exit() 
    
    dataGenObject=dataGen(filepath,noisePropogation)
    dataGenObject.dataGenerator()
