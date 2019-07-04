# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 13:54:41 2019

@author: safin
"""
import numpy as np


total_dollars = 0
N_sides = input("Enter a number for sides")
N_sides = int(N_sides)

x= []
for i in range(N_sides):
    x=np.append(x,i+1)
    
N_sides=x    


isBadside= input("isBadSide Vector")
isBadside = isBadside[1:-1].split(',')


for x in range(len(isBadside)):
    isBadside[x]=int(isBadside[x])

isBadside=np.asarray(isBadside)

countercounter = 0
for x in isBadside:
    if x == 0:
        isBadside[countercounter]=1
    else:
        isBadside[countercounter]=0
    countercounter= countercounter+1
    
Array_with_values = np.multiply(N_sides,isBadside)




def keepplay():
    global total_dollars
    
    random_value_from_array=np.random.choice(Array_with_values)
    if random_value_from_array == 0:
        print ('You lost!')
        total_dollars = 0
    else:
        total_dollars = total_dollars+ random_value_from_array
    
    
    

    
def quitplaying():
    return 0
     
## 1, 2 ,3
## 1 ,0  1

#
#total_dollars = 0
#epsilon = 1
#counter = 0
#while epsilon > .00001:
#    
#    value_of_current = (total_dollars + 
#                        (np.count_nonzero(Array_with_values)/len(Array_with_values))*sum(Array_with_values)/np.count_nonzero(Array_with_values) - 
#                        (1 - (np.count_nonzero(Array_with_values)/len(Array_with_values)))*(total_dollars+(np.count_nonzero(Array_with_values)/len(Array_with_values))*sum(Array_with_values)/np.count_nonzero(Array_with_values)) )
#    epsilon =  value_of_current - total_dollars
#    print(value_of_current)
#    total_dollars= value_of_current
#    counter= counter+ 1
#    
#print('expected return is ', total_dollars)
#    


    

total_dollars = -1
epsilon = 1000
counter = 0
number_of_nonzeros = np.count_nonzero(Array_with_values)
number_of_zeros=  len(Array_with_values) - number_of_nonzeros 
average_non_zero_values = sum(Array_with_values)/number_of_nonzeros
probability_of_winning = number_of_nonzeros/ len(Array_with_values)
probability_of_losing = 1 - probability_of_winning
while epsilon > .00001:
    
    value_of_current = (total_dollars + 
                        probability_of_winning*average_non_zero_values -
                        probability_of_losing*(total_dollars+ probability_of_winning*average_non_zero_values ))
                        
                        

                        
                        
    epsilon =  value_of_current - total_dollars
    print(value_of_current)
    total_dollars= value_of_current
    counter= counter+ 1
    
print('expected return is ', total_dollars)
    
    
    
    
    
    

    
    
