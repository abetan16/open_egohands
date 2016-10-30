"""

"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"

from abc import ABCMeta, abstractmethod
from egovision.handSegmentation import SegmentVisualizer
from egovision import Frame
from egovision.interfaces import cv2i
from egovision.values.colors import HAND_COLOR
import numpy as np

class AbstractOcclusionDetector(object):
    __metaclass__ = ABCMeta
    


    @abstractmethod
    def getOcclusionState(self):
        pass

    @abstractmethod
    def splitOcclusion(self):
        pass

    def visualize(self):
        def overlapSuperpixelBoundaries(frame, slic, leftMask, rightMask):
            compressedFrame = self.frame.resizeByWidth(self.compressionWidth)
            if leftMask:
                compressedFrame = SegmentVisualizer.__overlapSegmentation__(compressedFrame, leftMask,
                                                              HAND_COLOR[0])
            if rightMask:
                compressedFrame = SegmentVisualizer.__overlapSegmentation__(compressedFrame, rightMask,
                                                              HAND_COLOR[1])
            boundaryMask = self.superpixelAlgorithm.boundaryMask(slic)
            compressedFrame = Frame(cv2i.addWeighted(compressedFrame.matrix, 1, boundaryMask.matrix, 1, 0))
            return compressedFrame

        if self.tempSuperpixelAlgorithm and self.others["split"]:
            compressedFrame = self.tempFrame.resizeByWidth(self.compressionWidth)
            occludedMask = SegmentVisualizer.__overlapSegmentation__(compressedFrame, self.tempOccludedMask,
                                                                     HAND_COLOR[0])
            previousFrame = overlapSuperpixelBoundaries(self.frame, self.superpixelAlgorithm.lastLabels, self.leftMask, self.rightMask)
            currentFrame = overlapSuperpixelBoundaries(self.tempFrame, self.tempSuperpixelAlgorithm.lastLabels, self.tempLeftMask, self.tempRightMask)

            visualFrame = np.hstack((occludedMask.matrix, previousFrame.matrix, currentFrame.matrix))
            
            SegmentVisualizer.showFrameQuick("current Slic",Frame(visualFrame))
            

            
