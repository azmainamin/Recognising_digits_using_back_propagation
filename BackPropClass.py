#!/usr/bin/env python
# BackPropagation class
# By Azmain Amin



from __future__ import division
from __future__ import print_function
import numpy as np
import pickle



class BackPropagation(object):

    def __init__(self,n,m,h,w = 1.0):
        """ Constructor for the BackProp class. Takes the number of inputs, outputs, and hidden untis as parameters."""
        self.inp = n
        self.out = m
        self.h = h
        # Set up --> Randomly choosen weights that are close to 0
        self.Wih = np.random.randn(n+1,h) * w
        self.Who = np.random.randn(h+1,m) * w

    def __str__(self):
        """ Returns the string representation of the BackProp object"""
        return ("This is a perceptron with %d input(s),%d output(s), and %d hidden units." % (self.inp, self.out, self.h))


    def test(self, I):
        """ Returns the output as a float value."""
        Hnet = np.dot(np.append(I,1), self.Wih)
        H = self.squashFunc(Hnet)
        Onet = np.dot(np.append(H,1),self.Who)
        O = self.squashFunc(Onet)

        return O
    
    def train(self, inp_pat, tar_pat, niter = 10000, eta = 0.5):
        """Takes the input, target, number of iterations and the learning rate as input. Returns two matrix of weights: Weights from input to hidden units and weights from hidden units to output."""
        
        p = len(inp_pat)
        
        for i in range(niter):
            print("Completed: ",i+1,"/",niter, "times.") 
            deltaWih = np.zeros(self.Wih.shape)
            deltaWho = np.zeros(self.Who.shape)
            for j in range(p):
                Hnet = np.dot(np.append(inp_pat[j],1), self.Wih)
                H = self.squashFunc(Hnet)
                Onet = np.dot(np.append(H,1),self.Who)
                O = self.squashFunc(Onet)
                dO = (tar_pat[j] - O) * (O*(1-O))
                dH = np.dot(dO,self.Who.T)[:-1] * (H*(1-H))
                deltaWih += np.outer(np.append(inp_pat[j],1),dH)
                deltaWho += np.outer(np.append(H,1), dO)

            self.Wih += eta * deltaWih
            self.Who += eta * deltaWho

        return self.Wih, self.Who
            

    def squashFunc(self, f):
        """ Returns the squashed value of the function."""
        return (1/(1+np.exp(-f)))

    def save(self,filename, protocol = 3):
        """ Saves the object in a binary file.If you are using Python2, use protocol = 2"""    
        pickle.dump(self, open(filename, "wb"),protocol )


    def load(self, filename):
        """ Loads the object in a binary file."""  
        fileObj = open(filename, "rb")
        bp = pickle.load(fileObj)
        return bp


        
                               
                    



