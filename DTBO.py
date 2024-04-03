# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 11:47:41 2023

@author: Payam_(cyrus)
"""

import numpy as np
from numpy.random import rand as r
from numpy.random import randint,random
import matplotlib
from matplotlib import pyplot as plt

# x=8
# y=9
# z=3
dimension=2
SearchAgents=10
Max_iterations=500
lowerbound=[-500,0]
upperbound=[10,150]

# def fitness (x,y):
#     fit=np.cos(x)+np.sin(y)
#     return fit

# def fitness (x,y,z): #my_fit
#     fit=np.cos(k[0])+np.sin(k[1])*(np.tan(k[2]))
#     return fit

# def fitness (x,y):   #F17
#     F17=(y-(x**2)*5.1/(4*(np.pi**2))+5/np.pi*x-6)**2+10*(1-1/(8*np.pi))*np.cos(x)+10
#     return F17

# def fitness (x,y):   # booth
#     F17=(x + 2*y - 7)**2 + (2*x + y - 5)**2
#     return F17

# def fitness (x,y):   #bukin
#     F17=100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)
#     return F17

def fitness (x,y):   #matyas
    F17=0.26 * (x**2 + y**2) - 0.48 * x * y
    return F17

# def fitness (x,y):   #F
#     F=(2*(x**2))+3*(y**2)
#     return F

space = np.linspace(-50, 50, 100)
X1,Y1 = np.meshgrid(space,space)
fitness_plot=fitness(X1,Y1)
# np.min(fitness_plot)
figu=plt.figure()
ax=plt.axes(projection='3d')
ax.plot_surface(X1,Y1,fitness_plot
                ,cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.plot

# def DTBO (SearchAgents=30,Max_iterations,lowerbound,upperbound,dimension,fitness):

X=np.float64(np.random.randint(lowerbound,upperbound,(SearchAgents,dimension)))

lowerbound=np.ones((1,dimension))*lowerbound         #Lower limit for variables
upperbound=np.ones((1,dimension))*upperbound         #Upper limit for variables

best_so_far=[]

for t in range(0,Max_iterations):
    
        XF=[]
        fitnes_0=[]
        counter=0
        for k in X:
            fit=fitness(k[0],k[1])
            counter+=1
            #print(fit)
            fitnes_0.append([fit,counter])
            XF.append([k,fit])
        fitnes_0.sort()
        
        def takeSecond(elem):
            return elem[1]
        XF.sort(key=takeSecond)
        
        # update the best member
        best = XF[0]
        if t == 0:
            Xbest = XF[0][0]                        # Optimal location
            fbest=XF[0][1]                          #The optimization objective function
            
        elif best[1] <= fbest :
            fbest=best[1]
            Xbest=best[0][1]                       # Optimal location
                 
        
    # t=1; 
        N_DI=1+round(0.2*SearchAgents*(1-t/Max_iterations))
        # DI_l=int(N_DI*len(fitnes_0))
        DI=XF[:N_DI]
        
        #update DTBO population
        for i in range(0,SearchAgents):
            
            # Phase 1: training by the driving instructor (exploration)
            k_i=randint(0,N_DI) ######################################
            DI_ki=DI[k_i]
            F_DI_ki=DI[k_i][1]
            I=round(1+randint(0,2))
            if F_DI_ki < fitnes_0[i][0]:
                X_PI=X[i]+r(1)[0] * (DI_ki[0][0] - (I * X[i]))  # Eq(5)
            else:
                X_PI=X[i]+r(1)[0] * (1* X[i] - I * DI_ki[0][0]) # Eq(5)
             
            # /update X_I based on Eq(6)
            F_P1 = fitness(X_PI[0],X_PI[1])
            if F_P1 <= fitnes_0[i][0]:
                X[i]=X_PI
                
          #  %% END Phase 1: training by the driving instructor (exploration)
    
          #  %% Phase 2: learner driver patterning from instructor skills (exploration)
            
            PIndex=0.01+0.9*(1-t/Max_iterations)
            X_P2=(PIndex)* X[i]+(1-PIndex) * (Xbest)  # Eq. (7)

            X_P2= max(tuple(X_P2),tuple(lowerbound[0]))
            X_P2 = min(tuple(X_P2),tuple(upperbound[0]))
    
         #         Update X_i based on Eq(8)
            F_P2 = fitness(X_P2[0],X_P2[1])
            if F_P2 <= fitnes_0[i][0]:
                X[i] = X_P2;
                fitnes_0[i][0]=F_P2
  
            # END Phase 2: learner driver patterning from instructor skills (exploration)
            
            # Phase 3: personal practice (exploitation)
            R=0.05
            X_P3= X[i]+ (1-2*random((1,dimension)))[0]*R*(1-t/Max_iterations)*X[i] # Eq.(9)
            X_P3= max(tuple(X_P3),tuple(lowerbound[0]))
            X_P3= max(tuple(X_P3),tuple(upperbound[0]))
            
            # Update X_i based on Eq(10)
            F_P3 = fitness(X_P3[0],X_P3[1])
            if F_P3 <= fitnes_0[i][0]:
                X[i] = X_P3;
                fitnes_0[i][0]=F_P3
          
            # END Phase 3: personal practice (exploitation)
            
        best_so_far.append(fbest)
    
Best_score=fbest
Best_pos=Xbest
DTBO_curve=best_so_far;

plt.figure()
plt.axes()
t=[]
for i in range(0,Max_iterations):t.append(i)
plt.plot(t, best_so_far)
plt.ylabel('best solution')
plt.xlabel('Iteration')
plt.show()
print(fbest)















    
    