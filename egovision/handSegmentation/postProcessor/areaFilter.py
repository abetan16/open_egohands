""" 

"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"


from egovision.interfaces import cv2i
from abstractPostProcessor import AbstractPostProcessor
import numpy as np

class AreaFilter(AbstractPostProcessor):
    
    def __init__(self, compressionWidth = 200, minAreaPercentage = 0.01):
        AbstractPostProcessor.__init__(self, np.ndarray, np.ndarray, compressionWidth)
        self.minAreaPercentage = minAreaPercentage
        self.output = None
        self.others = {}

    def process(self, contours, frameShape):
        self.frameArea = frameShape[0]*frameShape[1]
        self.minSize = self.frameArea*self.minAreaPercentage
        self.input = contours
        discartedSize = filter(lambda x: cv2i.contourArea(x) < self.minSize, contours)
        contours = filter(lambda x: cv2i.contourArea(x) >= self.minSize, contours)
        self.output = contours
        self.others["discarted"] = discartedSize
        return contours

