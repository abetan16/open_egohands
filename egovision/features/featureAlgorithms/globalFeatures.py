__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['GIST','ColorHistogram']
from featureAlgorithm import FeatureAlgorithm

try:
    import leargist
    from PIL import Image
except:
    import warnings
    warnings.warn("GIST is not available",ImportWarning)
#from utilities import compressFrame
import sys,os
sys.path.append(os.path.abspath('../../../'))
from egovision.interfaces import cv2i
import numpy

RANGES = {
    "HSV-HIST": numpy.array([181,256,256]),
    "RGB-HIST": numpy.array([256,256,256]),
    "LAB-HIST": numpy.array([256,256,256]),
    "YCrCb-HIST": numpy.array([256,256,256])
}

NBINS = {
    "HSV-HIST": numpy.array([6,4,4]),
    "RGB-HIST": numpy.array([4,4,4]),
    "LAB-HIST": numpy.array([4,4,4]),
    "YCrCb-HIST": numpy.array([4,4,4])
}

class GIST(FeatureAlgorithm):

    def get(self, frame):
        img = Image.fromarray(frame)
        desc = leargist.color_gist(img)
        return desc

class ColorHistogram(FeatureAlgorithm):
    
    def __init__(self, feature):
        self.feature = feature

    def get(self, frame):
        if self.feature == "HSV-HIST":
            frame = cv2i.cvtColor(frame, cv2i.COLOR_BGR2HSV)
        elif self.feature == "LAB-HIST":
            frame = cv2i.cvtColor(frame, cv2i.COLOR_BGR2LAB)
        elif self.feature == "YCrCb-HIST":
            frame = cv2i.cvtColor(frame, cv2i.cv.CV_BGR2YCrCb)

        ranges = RANGES[self.feature]
        nbins = NBINS[self.feature]
        matrixHIST = frame
        hist = numpy.zeros(tuple(nbins))
        step = numpy.array(ranges)/numpy.array(nbins)
        desc = matrixHIST.reshape(matrixHIST.size/3, 3)
        desc = desc[numpy.random.randint(0,len(desc),1000)]
        for pix in desc:
            pix = pix/step
            hist[pix[0]][pix[1]][pix[2]] += 1
        return hist.flatten()/1000
