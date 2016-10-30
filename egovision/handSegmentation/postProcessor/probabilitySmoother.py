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
import copy
import numpy as np

class ProbabilitySmoother(AbstractPostProcessor):
    
    def __init__(self, compressionWidth = 200, sigma=(5,5)):
        AbstractPostProcessor.__init__(self, Frame, Frame, compressionWidth)
        self.sigma = sigma
    
    def process(self, frame):
        blured = cv2i.GaussianBlur(frame.matrix, self.sigma, 0, 0, cv2i.BORDER_REFLECT)
        blured = cv2i.normalize(blured, alpha=0, beta=np.max(frame.matrix), norm_type=cv2i.NORM_MINMAX)
        blured = blured.reshape(frame.matrix.shape)
        self.output = copy.deepcopy(Frame(blured))
        self.input = copy.deepcopy(Frame(frame.matrix))

        return Frame(blured)
     



