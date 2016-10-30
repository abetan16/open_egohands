""" 

"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"

from egovision import Frame
from egovision.interfaces import cv2i
from abstractPostProcessor import AbstractPostProcessor
import numpy as np
import copy

class ProbabilityThresholder(AbstractPostProcessor):
    
    def __init__(self, compressionWidth = 200, threshold = 0.4):
        AbstractPostProcessor.__init__(self, Frame, np.ndarray, compressionWidth)
        self.threshold = threshold

    def process(self, probabilityMap):
        self.input = copy.deepcopy(probabilityMap)       
        probabilityMap.matrix[probabilityMap.matrix >= self.threshold] = int(1)
        probabilityMap.matrix[probabilityMap.matrix < self.threshold] = int(0)
        probabilityMap.matrix = probabilityMap.matrix.astype(np.uint8)
        contours = cv2i.findContours(probabilityMap.matrix, cv2i.RETR_EXTERNAL, cv2i.CHAIN_APPROX_NONE)[0]    
        contours.sort(key=lambda x: cv2i.contourArea(x),reverse=True)
        self.output = contours
        return contours
