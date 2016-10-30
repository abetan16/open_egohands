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

class NumberFilter(AbstractPostProcessor):
    
    def __init__(self, compressionWidth = 200, maxNumber = 3):
        AbstractPostProcessor.__init__(self, np.ndarray, np.ndarray, compressionWidth)
        self.maxNumber = maxNumber
        self.output = None
        self.others = {}

    def process(self, contours):
        self.input = contours
        contours.sort(key=lambda x: cv2i.contourArea(x),reverse=True)
        discarted2 = []
        if len(contours) >= self.maxNumber:
            discarted2 = contours[self.maxNumber:]
            contours = contours[:self.maxNumber]
        self.output = contours
        self.others["discarted"] = discarted2
        return contours


