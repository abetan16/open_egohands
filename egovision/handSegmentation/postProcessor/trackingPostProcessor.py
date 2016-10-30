""" 

"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"

from contourThresholder import ProbabilityThresholder
from probabilitySmoother import ProbabilitySmoother
from abstractPostProcessor import AbstractPostProcessor
from egovision import Frame
import numpy as np
from egovision.output import VideoVisualizer
from egovision.interfaces import cv2i

class TrackingPostProcessor(AbstractPostProcessor):
    """ 

    """

    def __init__(self, sigma = (3,3),
                       probabilityThreshold = 0.4,
                       compressionWidth = 200):
        AbstractPostProcessor.__init__(self, Frame, Frame, compressionWidth)
        self.smoother = ProbabilitySmoother(compressionWidth, sigma)
        self.thresholder = ProbabilityThresholder(compressionWidth, probabilityThreshold)
        self.output = None
        self.others = {}


    def process(self, frame, targetShape=None):
        self.input = frame
        smoothedMap = self.smoother.process(frame) # Frame -> Frame
        contours = self.thresholder.process(smoothedMap) # Frame -> Contours
        contours = filter(lambda x: len(x) > 5, contours)
        self.output = contours
        segment = self.__getOutputFrame__(frame, targetShape)
        segment.matrix /= 255
        self.others["smoothedMap"] = smoothedMap 
        self.others["contours"] = contours
        return segment
