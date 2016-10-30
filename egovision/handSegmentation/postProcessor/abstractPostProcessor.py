""" 

"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['HandSegmenter']

from abc import ABCMeta, abstractmethod
from egovision.output import VideoVisualizer
from egovision import Frame
from egovision.interfaces import cv2i
import numpy as np

class AbstractPostProcessor(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, inputType, outputType, compressionWidth):
        self.inputType = inputType
        self.outputType = outputType
        self.compressionWidth = compressionWidth
        self.input = None
        self.output = None
        self.others = {}

    @abstractmethod
    def process(self, *args, **kargs):
        pass

    def __getFrameMatrix__(self, attribute, frame, targetShape = None):
        if targetShape is None:
            targetShape = frame.matrix.shape

        scalingFactor = targetShape[1]/float(self.compressionWidth)
        
        if isinstance(attribute, Frame):
            result = attribute.resizeByWidth(frame.matrix.shape[1]*scalingFactor)
            result = result.matrix
        elif isinstance(attribute, np.ndarray) or isinstance(attribute, list):
            if scalingFactor != 1.0:
                scaledContours = map(lambda x: (scalingFactor*x).astype("int32"), attribute)
            else:
                scaledContours = attribute
            result = cv2i.contours2binary(scaledContours, targetShape)*255
            result = result.astype(np.uint8)
        return result
       
    def __getOutputFrame__(self, frame, targetShape):
        return Frame(self.__getFrameMatrix__(self.output, frame, targetShape))

    def visualize(self, frame):
        visualizer = VideoVisualizer()
        steps = []
        steps.append(self.__getFrameMatrix__(self.input, frame))
        output = self.__getFrameMatrix__(self.output, frame)
        steps.append(output)
        visualizeFrame = Frame(np.hstack(steps))
        visualizer.showFrameQuick(type(self).__name__, visualizeFrame)
        for i in range(1000):
            cv2i.waitKey(1)

