#!/usr/bin/env python
# This program trains a BackProp to learn recognize digits
# By Azmain Amin

from BackPropClass import *
from DigitClass import *
from makeDigits import *
import numpy as np

# List of patterns for training
def train(inputs, targets):
    """Trains a backprop to learn digit recognition."""
    
    bp = BackPropagation(196,10, 15,0.1)

    #To pickle, uncomment the train and save lines. When unpickling, comment them out.
    #If you are using Python2, pass in protocol = 2 in bp.save()

    #wih, who = bp.train(inputs, targets,8000,0.01)
    #bp.save("temp_weights.dat")  

    return bp

def makeInputAndTarget():
    """Makes training inputs and targets from digit_train.txt"""

    patterns = read_data("digits_train.txt")

    # Make the matrices
    inputs = np.empty((2500,196))
    targets = np.zeros((2500, 10))

    #Making Targets
    for row in range(2500):
        inputs[row] = patterns[row].getInput()
        col = row//250
        targets[row][col] = 1

    return inputs, targets


def makeTestInput():
    """Makes training inputs from digit_test.txt."""

    pattern_test = read_data("digits_test.txt")
    inputs2 = np.empty((2500,196))

    for j in range(2500):
        inputs2[j] = pattern_test[j].getInput()

    return inputs2

def displayOutput(bp,inputs2):

    confusion = np.zeros((10,10), "int")

    #When pickling, comment the load line and while unpickling, remove the comment
    bp = bp.load("part3_weights.dat")
    for k in range(2500):
        row = k//250
        col = np.argmax(bp.test(inputs2[k]))
        confusion[row][col] += 1

    print("-" * 25 + "PART THREE" + "-" * 35)
    print("%5s%4s%4s%4s%4s%4s%4s%4s%4s%4s" % ("0","1","2","3","4","5","6","7","8","9"))
    print("-" * 70)
    print(confusion)
    print("-" * 69)    

def main():
    inputs, targets = makeInputAndTarget()
    bp = train(inputs, targets)
    inputs_test = makeTestInput()
    displayOutput(bp,inputs_test)
    
    
if __name__ == "__main__":
    main()

