__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['OpenCvInterface']

import sys,os
sys.path.append(os.path.abspath('../../'))
import cv2
import numpy as np


class OpenCvInterface:

    def contours2binary(self, contours, frameShape):
        segment = np.zeros(frameShape)
        for nc, co in enumerate(contours):
            cv2.drawContours(segment, contours, nc, (255),-1)
        segment[segment>=0.1] = int(1)
        return segment.astype(np.uint8)
    
    def VideoCapture(self, *args, **kargs):
        return cv2.VideoCapture(*args, **kargs)

    def imread(self, *args, **kargs):
        return cv2.imread(*args, **kargs)

    def normalize(self, *args, **kargs):
        return cv2.normalize(*args, **kargs)

    def imwrite(self, *args, **kargs):
        return cv2.imwrite(*args, **kargs)

    def resize(self, *args, **kargs):
        return cv2.resize(*args, **kargs)

    def cvtColor(self, *args, **kargs):
        return cv2.cvtColor(*args, **kargs)

    def threshold(self, *args, **kargs):
        return cv2.threshold(*args, **kargs)

    def boundingRect(self, *args, **kargs):
        return cv2.boundingRect(*args, **kargs)

    def contourArea(self, *args, **kargs):
        return cv2.contourArea(*args, **kargs)

    def fitEllipse(self, *args, **kargs):
        return cv2.fitEllipse(*args, **kargs)

    def rectangle(self, *args, **kargs):
        return cv2.rectangle(*args, **kargs)

    def ellipse(self, *args, **kargs):
        return cv2.ellipse(*args, **kargs)

    def circle(self, *args, **kargs):
        return cv2.circle(*args, **kargs)

    def addWeighted(self, *args, **kargs):
        return cv2.addWeighted(*args, **kargs)

    def putText(self, *args, **kargs):
        return cv2.putText(*args, **kargs)

    def waitKey(self, *args, **kargs):
        return cv2.waitKey(*args, **kargs)

    def imshow(self, *args, **kargs):
        return cv2.imshow(*args, **kargs)

    def destroyWindow(self, *args, **kargs):
        return cv2.destroyWindow(*args, **kargs)

    def __getattr__(self, name):
        return cv2.__getattribute__(name)


cv2i = OpenCvInterface()
