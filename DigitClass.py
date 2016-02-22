# This program reads in the train and test.txt and creates a Digit object.
# By Azmain Amin

from __future__ import division
from __future__ import print_function
import numpy as np
import re

class Digit(object):

    def __init__(self, I):
        self.tmp_block = I[1:len(I)-1] 
        tmp_input = "".join(self.tmp_block).split()
        self.tmp_target = I[-1]

        self.inputs = np.array([float(i) for i in tmp_input])
        self.target = np.array([self.tmp_target])
 
    def getInput(self):
        return self.inputs

    def getTarget(self):
        return self.target

    def __str__(self):
        for i in range(len(self.tmp_block)):
            self.tmp_block[i] += "\n"
        self.tmp_block = "".join(self.tmp_block).replace(".0"," ")
        self.tmp_block = re.sub(r'[^ \n]',"*",self.tmp_block)
        

        self.tmp_target2 = self.tmp_target.replace("0","")
        for i in range(len(self.tmp_target2)):
            
            if self.tmp_target2[i] == '1':
                target = i
           
    
        return ("Input: \n" + self.tmp_block +"\nTarget: " + str(target) + "\n")

        
