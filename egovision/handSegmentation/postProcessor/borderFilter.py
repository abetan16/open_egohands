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

class BorderFilter(AbstractPostProcessor):
    
    def __init__(self, compressionWidth = 200, marginPercentage = 0.05):
        AbstractPostProcessor.__init__(self, np.ndarray, np.ndarray, compressionWidth)
        self.marginPercentage = marginPercentage

    def process(self, contours, frameShape):
        self.width = frameShape[1]
        self.height = frameShape[0]
        self.input = contours
        margin = self.width*self.marginPercentage # ten percent of the width
        temp = []
        discarted = []
        for nc, contour in enumerate(contours):
            xmin, ymax, width, height = cv2i.boundingRect(contour)
            xmax = xmin + width
            ymin = ymax + height
            if len(contour) > 3:
                if margin > xmin:
                    temp.append(contour)
                elif self.width - margin < xmax: # interects the sides
                    temp.append(contour)
                elif self.height - margin < ymin:
                    temp.append(contour)
                else:
                    discarted.append(contour)
            else:
                discarted.append(contour)
        contours = temp
        self.output = contours
        self.others["discarted"] = discarted
        return contours


