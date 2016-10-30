"""
EgoVision: Library to process Egocentric Videos.
"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"

def getPrecision(cm):
    return cm[0,0]/float(cm[0,0]+cm[1,0]), cm[1,1]/float(cm[1,1]+cm[0,1]) 

def getRecall(cm):
    return cm[0,0]/float(cm[0,0] + cm[0,1]), cm[1,1]/float(cm[1,1]+cm[1,0])

def getFScore(cm):
    p0, p1 = getPrecision(cm)
    r0, r1 = getRecall(cm)
    f0 = 2*p0*r0/(p0+r0)
    f1 = 2*p1*r1/(p1+r1)
    return f0, f1

