""" In seek of simplicity we define a handSegmenter as a family of objects that
are able to perform the following operations: i) Being trained
[trainClassifier] ii) Segment a new frame [segmentFrame] iii) postProcess the
frame to remove some false positives.
    
In general depending of the particular algorithm the arguments of the
trainClassifier could slightly change and must be redefined in the particular
inherited handSegmenter class. As an example the PixelByPixelHandSegmenter
algorithm is trained exclusively with the mask filename, while the
PixelByPixelMultiHandSegmenter requires a list of masks filenames and a
modelRecommender. The segmentFrame method must be written to reflect the
segmentation algorithm. 

In summary, a pixel by pixel hand segmenter is relies on a classification
strategy at a pixel level. In the training phase the segmenter is fitted to
classifty the color components of a pixel as hand-like(1) or background(0). In
the testing face the fitted classifier is applied to each pixel to delineate
the hands of the user.

Example 1: Creating a pixel by pixel hand-segmenter::

    from egovision import Frame
    from egovision.handSegmentation import PixelByPixelHandSegmenter

    # Just one mask and one frame
    datasetFolder = "egovision/dataExamples/GTEA/"
    mask = "".join([datasetFolder,"masks/GTEA_S1_Coffee_C1/00000780.jpg"])
    frameFile = "".join([datasetFolder,"img/GTEA_S1_Coffee_C1/00000780.jpg"])


    # TRAINING PHASE (FEATURE, COMPRESSIONWIDTH, CLASSIFIERTYPE)
    hs = PixelByPixelHandSegmenter("LAB", 200, "RF")
    dataManager = hs.trainClassifier([mask])


    # TESTING PHASE [JUST FOR ILLUSTRATIVE PURPOSES IN THE SAME TRAINING]
    frame = Frame.fromFile(frameFile)
    segment = hs.segmentFrame(frame)


Example 2: Creating a Multipixel with 20 illumination models (N) and 5 active
models (K)::

    from egovision import Frame
    from egovision.handSegmentation import PixelByPixelHandSegmenter
    from egovision.modelRecommenders import NearestNeighbors
    import os

    # MASK and FRAME folders must have the same files
    datasetFolder = "egovision/dataExamples/GTEA/"
    maskFolder = "".join([datasetFolder,"masks/GTEA_S1_Coffee_C1/"])
    allMasks = [maskFolder + x for x in os.listdir(maskFolder)]
    frameFolder = "".join([datasetFolder,"img/GTEA_S1_Coffee_C1/"])
    allFrames = [frameFolder + x for x in os.listdir(frameFolder)]


    # MODEL RECOMMENDER SYSTEM (nAverage, CompressionWidth, globalFeature)
    modelRecommender = NearestNeighbors(20, 200, "HSV-HIST")
    modelRecommender.train(allFrames[:5]) # 5 active models


    # TRAINING PHASE (FEATURE, COMPRESSIONWIDTH, CLASSIFIERTYPE)
    hs = PixelByPixelHandSegmenter("LAB", 200, "RF")
    dataManager = hs.trainClassifier(allMasks[:20]) # 20 illumination Models


    # TESTING PHASE [JUST FOR ILLUSTRATIVE PURPOSES IN THE SAME TRAINING]
    frameFile = "".join([datasetFolder,"img/GTEA_S1_Coffee_C1/00000780.jpg"])
    frame = Frame.fromFile(frameFile)
    segment = hs.segmentFrame(frame)

"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['HandSegmenter']

from abc import ABCMeta, abstractmethod
from egovision.extras import Sampler
from egovision.features import FeatureController
from egovision.interfaces import cv2i
import numpy as np


class HandSegmenter(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, feature, compressionWidth, classifier, step=3):
        self.feature = feature
        self.compressionWidth = compressionWidth
        self.featureController = FeatureController(compressionWidth, feature)
        self.classifierType = classifier
        self.sampler = None
        self.featureLength = None
        self.step = step
        self.TAG_LOG = "[Hand Segmenter] "
        self.GPU = False

    @abstractmethod
    def trainClassifier(self, *args, **kargs):
        pass

    @abstractmethod
    def segmentFrame(self,frame):
        pass

    def asEllipse(self, contour):
        ellipse = cv2i.fitEllipse(contour)
        return ellipse
    
    def preProcessData(self, descSampled):
        if self.GPU:
            from pycuda import gpuarray
            X = gpuarray.to_gpu(descSampled.astype(np.int32))
            X.smp_width = self.sampler.rows.shape[1]
            X.smp_height = self.sampler.rows.shape[0]
            return X
        else:
            return descSampled


    def postProcessPrediction(self, result):
        if self.GPU:
            return result.get_async()
        else:
            return result


