# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 11:25:10 2019

@author: safin
"""
def create_a_dataset():
    statehistory_vector =[3]
    samples_of_winnins=[]
    for k in range (100):
        
        set_of_ten = []
        
        for s in range(10):
            
            currentStates = [0,0,0,1,0,0,0]
            statehistory_vector =[3]
            for i in range(1000):
                
                
                action=rand.choices(['right','left'])
                if action[0] =='left':
                    
                    
                    value = currentstate(currentStates) -1
                    currentStates=[0,0,0,0,0,0,0]
                    currentStates[value]=1
                    #print(currentStates)
                    
                if action[0] =='right':
                    
                    value = currentstate(currentStates) +1
                    currentStates=[0,0,0,0,0,0,0]
                    currentStates[value]=1
                    #print(currentStates)
                statehistory_vector.append(value)
                if value==0:
                    #print("left won")
                    break
                if value==6:
                    #print("right one")
                    break
            set_of_ten.append(statehistory_vector)
            #samples_of_winnins.append(statehistory_vector)
        samples_of_winnins.append(set_of_ten)
    return samples_of_winnins
def currentstate(vector):
    
    for i in range(len(vector)):
        if i== 1:
            return vector.index(i)
def P_t(weight,x_t):
    
    weight=np.array(weight)
    x_t=np.array(x_t)
    return np.dot(weight,x_t)

        
def into_unit_vector(value):
    if value == 0:
        print("it ended on the left")
        return 0
    if value == 6:
        print("ended on the right")
        return 1
    if value == 1:
        return np.array([1,0,0,0,0])
    if value == 2:
        return np.array([0,1,0,0,0])
    if value == 3:
        return np.array([0,0,1,0,0])
    if value == 4:
        return np.array([0,0,0,1,0])
    if value == 5:
        return np.array([0,0,0,0,1])
    
def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions-targets)**2))

import random as rand
#rand.seed(74837)
import numpy as np



ideal_prediction = np.array([1/6,1/3,1/2,2/3,5/6])
#rand_weight=np.array([rand.uniform(0, 1),rand.uniform(0, 1),rand.uniform(0, 1),rand.uniform(0, 1),rand.uniform(0, 1)])
rand_weight = np.array([0,0,0,0,0])
currnet_weight=rand_weight
rsme_vector_for_ploting=[]

lamda_vec = [0,.1,.3,.5,.7,.9,.99]
for lamda_v in lamda_vec:
    another_vec=[]
    for ii in range(1):
        
        
        #print(ii)

        samples_of_winnins= create_a_dataset()

        ##example 3 would get <0,0,1,0,0>

       
        #weight=np.array([.54,.56,.57,.52,.51])
        weight=rand_weight
        
        counter=0
        epsiolon=1
        # while abs(weights-w) < epsilon:
        while epsiolon >.001 :
            different_delta=[]
            for x in samples_of_winnins:
                
                
                for y in range(len(x)):
                    statehistory_vector=x[y]
                    t= 0
                    alpha=.01
                    
                    lamda = lamda_v
                    some_vec=[]
                    while True:
                    
                        # summation of gradient for a given sequence
                        sum_of_gradient_times_lamda=np.array([0,0,0,0,0])
                        for k in range(t+1):
                            
                            sum_of_gradient_times_lamda=sum_of_gradient_times_lamda+ (lamda**(t-k))*into_unit_vector(
                                    statehistory_vector[k])
                            ## this is for
                        if statehistory_vector[t+1]==6:
                            p_t2=1
                        elif statehistory_vector[t+1]==0:
                            p_t2=0
                        else:
                            p_t2=P_t(weight,into_unit_vector(statehistory_vector[t+1]))
            
                        p_t1= P_t(weight,into_unit_vector(statehistory_vector[t]))
                        alpha_times_weights = alpha*(p_t2 - p_t1)
            
                        some_vec.append(alpha_times_weights*sum_of_gradient_times_lamda)
                      
                        if statehistory_vector[t+1]==6 or statehistory_vector[t+1]==0:
                            

                            ##update W
                        
                            break
                    
                        t+=1
                            
                    different_delta.append(sum(some_vec))
                
            delta_w=sum(different_delta)/len(different_delta)
            new_w= weight+delta_w
            #print(new_w)
            epsiolon=abs(sum(weight)-sum(new_w))
        #    print(epsiolon)
            counter+=1
            weight=new_w
            
            
            if counter > 1000:
                break
        another_vec.append(weight)
        
    
    weight=sum(another_vec)/len(another_vec)
    weight = weight / np.linalg.norm(weight)

    ###repeated presentation8 training paradigm 
    
    
    rsme_vector_for_ploting.append(rmse(weight,ideal_prediction))
    print(weight)
    print("current weight is",currnet_weight)
    currnet_weight=rand_weight
    print(rmse(weight,ideal_prediction))
    

    
import matplotlib.pyplot as pl
y=rsme_vector_for_ploting
x=np.array(lamda_vec)
pl.plot(x,y)
pl.plot(x,y,'bo')








        