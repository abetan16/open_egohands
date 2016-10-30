"""
EgoVision: Library to process Egocentric Videos.
"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['Sampler']


import numpy

class Sampler:
    def __init__(self, height=None, width=None, step=None):
        self.height = height 
        self.width = width
        self.step = step
        self.samplingIndexes()

    def samplingIndexes(self):
        self.cols = cols = numpy.ones(((self.height-1)/self.step +1, 1))*numpy.arange(0, self.width, self.step)
        self.rows = rows = (numpy.ones(((self.width-1)/self.step +1, 1))*numpy.arange(0, self.height, self.step)).T
        self.indexes = (cols+rows*self.width).flatten().astype(int)
    
    def sampleDescriptor(self, desc, nFeatures):
        featureLength = len(desc)
        desc = desc.reshape(featureLength/nFeatures, nFeatures)
        if self.step != 1:
            descSampled = desc[self.indexes]
        else:
            descSampled = desc
        return descSampled 
