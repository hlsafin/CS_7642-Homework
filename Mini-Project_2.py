import numpy as np
import pandas as pd
import math 
from sklearn.model_selection import train_test_split





"""Safin Salih 2/14/2018 Mini-Project 2 """

df = pd.read_excel('titanic_data.xlsx', sheetname='titanic_data')

 

""" Load Data"""
X=df.iloc[:,1:]
Y=((df.iloc[:,0]))

""" SPLIT DATA into 80/20 """

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.80,shuffle=True)

"""Question a"""

initialB=[1,1,1,1,1,1]
def logB(initialB):
    
    asdf=0
        
    for i in range(887):
        yyy=df.iloc[i,0]
        bw=np.dot([initialB],df.iloc[i,1:])
        bww=math.exp(-bw)
        bwww=1/(1+bww)
        first=math.pow(bwww,yyy)
        sw=1-bwww
        second=math.pow(sw,1-yyy)
        
        
        if ((first*second)==0):
            continue
        loggg=math.log(first*second)
        
        
        asdf=loggg+asdf
        
        
        
    return -1*asdf    
print(logB(initialB))        
        
""" Question B"""        
def gradB(B):
    
    asdf=[0,0,0,0,0,0]
    for i in range(887):
        
        
        yyy=df.iloc[i,0]
        bw=np.dot([B],df.iloc[i,1:])
        bww=math.exp(-bw)
        bwww=1/(1+bww)
        bsa=(yyy - bwww)
        feature=df.iloc[i,1:]
        ksa=bsa*feature
        asdf=asdf+ksa
        
        
        
    return -1*asdf
        
    
print(gradB(initialB))   

   
eta=.001
tol=math.pow(10,-6)

""" Gradient Ascent"""
def gradientAsce(Y_train,X_train,eta,tol):
    
    
    initialx=10
    stepsize=.001
    tol=math.pow(10,-6)
    previous_step_size = initialx
    
    
    while previous_step_size > tol:
        prev_x=initialx
        initialx+= stepsize*gradB(prev_x)
        previous_step_size = abs(initialx - prev_x)
        
        
        
    print("The Max occurs  at %f" % initialx)
    
gradientAsce(Y_train,X_train,eta,tol)   





        
        
        
        







