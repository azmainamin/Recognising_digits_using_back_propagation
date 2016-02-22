#!/usr/bin/env python
# This program reads in the digits_train.txt and digits_test.txt and returns a list of 2500 Digit objects.
# By Azmain Amin

from __future__ import division
from __future__ import print_function
import numpy as np
from DigitClass import *

def read_data(filename):
    block = open(filename).read().split("\nt")

    listOfDigits = []

    for i in range(len(block)):
    #To get rid of the nasty trailing newline character at the end of the file.
        if i == 2499:
            a = block[i].split("\n")
            del a[-1]
            digit = Digit(a)
            listOfDigits.append(digit)
        else:   
            a = block[i].split("\n")
            digit = Digit(a)
            listOfDigits.append(digit)

    return listOfDigits

def main():
    # Testing
    print(read_data("digits_train.txt")[0])
    print(read_data("digits_train.txt")[265])
    print(read_data("digits_train.txt")[1865])
    print(read_data("digits_train.txt")[2499])


if __name__=="__main__":
    main()
