""" Basic Objects of egoVision
"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['YUV','LAB','RGB','RHL','HSV']


import sys,os
sys.path.append(os.path.abspath('../../../'))
from egovision.interfaces import cv2i
import numpy
from featureAlgorithm import FeatureAlgorithm

class YUV(FeatureAlgorithm):
    def get(frame): 
        frame = cv2i.cvtColor(frame, cv2i.COLOR_BGR2YUV)
        desc = frame.flatten()
        return desc
    
class LAB(FeatureAlgorithm):
    def get(self, frame):
        frame = cv2i.cvtColor(frame, cv2i.COLOR_BGR2LAB)
        desc = frame.flatten()
        return desc

class RGB(FeatureAlgorithm):
    def get(self, frame):
        frame = cv2i.cvtColor(frame, cv2i.COLOR_BGR2RGB)
        desc = frame.flatten()
        return desc

class RHL(FeatureAlgorithm):
    def get(self, frame):
        frameAux = cv2i.cvtColor(frame, cv2i.COLOR_BGR2RGB)
        desc = frameAux.flatten()
        frameAux = cv2i.cvtColor(frame, cv2i.COLOR_BGR2HSV)
        numpy.append(desc,frameAux.flatten())
        frameAux = cv2i.cvtColor(frame, cv2i.COLOR_BGR2LAB)
        numpy.append(desc,frameAux.flatten())
        return desc


class HSV(FeatureAlgorithm):
    def get(self, frame):
        frame = cv2i.cvtColor(frame, cv2i.COLOR_BGR2HSV)
        desc = frame.flatten()
        return desc
