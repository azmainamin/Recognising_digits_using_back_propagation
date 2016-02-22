#!/usr/bin/env python
# This program trains a BackProp to learn how to solve 'AND', "OR" and "XOR"
# By Azmain Amin


import numpy as np
from BackPropClass import *


def testBoolPerc(func,I,t):
    bp = BackPropagation(2,1,3)
    p = len(I)

    #Train the backProp

    wih, who = bp.train(I,t,1000)
	
    #To Pickle, uncomment the next line
    #bp.save("xor_weights_protocol2.dat", 2)
    
    #To Unpickle, uncomment the next line.
    #bp = bp.load("backPropWeights.dat")


    print("-" * 25 + "PART ONE" + "-" * 35)
    print("Function: " + func)
    for j in range(p):
        out = (bp.test(I[j]))
        print(I[j],"---->",out)
    print("-" * 69)
def main():
	#Input and target for "AND"
    i_and = np.array([[0,0],[0,1],[0,1],[1,1]])
    t_and = np.array([[0],[0],[0],[1]])
	
	#Input and target for "OR"
    i_or = np.array([[0,0],[0,1],[0,1],[1,1]])
    t_or = np.array([[0],[1],[1],[1]])
    
	#Input and target for "XOR"
    i_xor = np.array([[0,0],[0,1],[0,1],[1,1]])
    t_xor = np.array([[0],[1],[1],[0]])
    
   
	
    testBoolPerc("AND", i_and, t_and)
    testBoolPerc("OR", i_or, t_or)
    testBoolPerc("XOR", i_xor, t_xor)
          
if __name__ == "__main__":
    main()
