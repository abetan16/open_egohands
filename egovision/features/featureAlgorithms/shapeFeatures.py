__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['HOG']

import sys,os
sys.path.append(os.path.abspath('../../../'))
from egovision.interfaces import cv2i
import numpy as np
from featureAlgorithm import FeatureAlgorithm

class HOG(FeatureAlgorithm):
    def get(self, frame):
        blockSize = 16
        blockStride = 8
        offsetWidth = (frame.shape[1]-blockSize)%blockStride
        offsetHeight = (frame.shape[0]-blockSize)%blockStride
        frame = cv2i.resize(frame,(frame.shape[1]-offsetWidth,frame.shape[0]-offsetHeight))
        nbins = 9
        params = dict(
            #_winSize = (64,128),
            _winSize = (len(frame[0]),len(frame)),
            _blockSize = (blockSize,blockSize),
            _blockStride = (blockStride,blockStride),
            _cellSize = (blockStride,blockStride),
            _nbins = nbins, #Default = 9
            _derivAperture = 1,
            _winSigma = -1,
            _histogramNormType = cv2i.HOGDESCRIPTOR_L2HYS,
            _L2HysThreshold = 0.2,
            _gammaCorrection = False,
            _nlevels = cv2i.HOGDESCRIPTOR_DEFAULT_NLEVELS
        )
        hog = cv2i.HOGDescriptor(**params) 
        #hog =  cv2.HOGDescriptor() 
        #desc = hog.compute(frame, hog.blockStride,(30,30))
        desc = hog.compute(frame, (0,0),(0,0)) # Win_stride, panning
        desc = np.transpose(desc)[0]
        return desc

