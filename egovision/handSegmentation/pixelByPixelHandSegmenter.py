"""The simplest hand segmenter is the PixelByPixelHandSegmenter and consist of
a single classifier that is trained using just one training frame and its mask.
This classifier can be understood as a naive segmentation strategy without any
mechanism to deal with illumination changes.

.. image:: ../_images/diagrams/singlemodel.png
    :align: center

.. automethod:: egovision.handSegmentation.PixelByPixelHandSegmenter.trainClassifier

.. automethod:: egovision.handSegmentation.PixelByPixelHandSegmenter.segmentFrame
"""


__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['PixelByPixelHandSegmenter']

import sys, os
sys.path.append(os.path.abspath('../../'))
from egovision import Frame
import numpy as np
from egovision.interfaces import cv2i
from egovision.interfaces import sklearni 
import pyximport; pyximport.install()
from handSegmenter import HandSegmenter
from egovision.extras import Sampler
from dataManager import MaskBasedDataManager

class PixelByPixelHandSegmenter(HandSegmenter):
    """
    
    """    

    def __init__(self, feature, compressionWidth, classifier, step=3):
        HandSegmenter.__init__(self, feature, compressionWidth, classifier, step)
        if classifier == "SVM":
            self.classifier = sklearni.SVC(kernel="linear",probability=False)
        elif classifier == "RF":
            self.classifier = sklearni.RandomForestRegressor(max_depth=10, min_samples_split=10)        
        elif classifier == "RF_GPU":
            from gpuRandomForestRegressor import GPURandomForestRegressor, GPURandomForestRegressor
            self.classifier = GPURandomForestRegressor(max_depth=10, min_samples_split=10)        
            self.GPU = True
        elif classifier == "RF_GPU2":
            from gpuRandomForestRegressor import GPURandomForestRegressorV2
            self.classifier = GPURandomForestRegressorV2(max_depth=10, min_samples_split=10)        
            self.GPU = True
        else:
            from exceptions import UnavailableClassifier
            raise UnavailableClassifier("Sorry, {0} is not implemented yet!".format(classifier))
    
    def __getClassifier__(self, position=None):
        return self.classifier
    

    def trainClassifier(self, maskFiles):
        """ 
        This method train a pixel by pixel hand segmenter using the
        provided masksFiles. The trained model is kept as an attribute inside
        the handSegmenter and the datamanager is returned.
        
        :ivar List maskFiles: Filenames to train the single classifier
        
        :returns: [DataManager] Data manager used to train the model.

        Example 1: Training a single pixel by pixel hand-segmenter::

            from egovision.handSegmentation import PixelByPixelHandSegmenter
            from egovision.extras import ObjectPickler

            # Just one mask and one frame
            datasetFolder = "egovision/dataExamples/GTEA/"
            mask = "".join([datasetFolder,"masks/GTEA_S1_Coffee_C1/00000780.jpg"])

            # TRAINING PHASE (FEATURE, COMPRESSIONWIDTH, CLASSIFIERTYPE)
            hs = PixelByPixelHandSegmenter("LAB", 200, "RF")
            dataManager = hs.trainClassifier([mask])
                
            # SAVING THE HAND-SEGMENTER FOR FUTURE USES
            from egovision.extras import ObjectPickler
            ObjectPickler.save("handSegmenter_test.pk")
        
        Example 2: Training a single pixel by pixel hand-segmenter with multiple masks::

            from egovision.handSegmentation import PixelByPixelHandSegmenter

            from egovision.handSegmentation import PixelByPixelHandSegmenter
            from egovision.extras import ObjectPickler
            import os

            # get the name of 3 masks
            datasetFolder = "egovision/dataExamples/GTEA/"
            maskFolder = "".join([datasetFolder,"masks/GTEA_S1_Coffee_C1/"])
            allMasks = [maskFolder + x for x in os.listdir(maskFolder)]

            # TRAINING PHASE (FEATURE, COMPRESSIONWIDTH, CLASSIFIERTYPE)
            hs = PixelByPixelHandSegmenter("LAB", 200, "RF")
            dataManager = hs.trainClassifier(allMasks[0:3]) # Use 3 masks
                
        Example 3: Loading the previously saved hand-segmenter::

            from egovision.handSegmentation import PixelByPixelHandSegmenter
            from egovision.extras import ObjectPickler

            # LOADING THE HAND-SEGMENTER
            hs = ObjectPickler.load(PixelByPixelHandSegmenter,"handSegmenter_test.pk")

        """
        dataManager = MaskBasedDataManager()
        dataManager.readDataset(maskFiles, None, self.feature)
        self.classifier.fit(dataManager.attributes,dataManager.categories)    
        self.featureLength = dataManager.attributes.shape[1]
        return [dataManager]

    def segmentFrame(self, frame):
        """ Once the model is trained the next step is to use on new frames.
        This method applies the single pixel by pixel hand segmenter on a
        particular frame object.  

        :ivar Frame frame: frame to be segmented
        
        :returns: [Frame] probability map with the skin-like likehood. The \
        returned frame has the compressionWidth to simplify its use on other \
        parts of this library. Be careful this is not the binary mask a \
        continous probability map.

        Example 1: Loading and using a previously saved hand segmenter::

            from egovision import Frame
            from egovision.handSegmentation import PixelByPixelHandSegmenter
            from egovision.extras import ObjectPickler


            # Just one mask and one frame
            datasetFolder = "egovision/dataExamples/GTEA/"
            frameFile = "".join([datasetFolder,"img/GTEA_S1_Coffee_C1/00000780.jpg"])


            # LOADING THE HAND-SEGMENTER
            hs = ObjectPickler.load(PixelByPixelHandSegmenter,"handSegmenter_test.pk")


            # SEGMENTING A NEW FRAME
            frame = Frame.fromFile(frameFile)
            segment = hs.segmentFrame(frame)            

        """

        # GET THE FEATURES
        success, featureVideo = self.featureController.getFeatures(frame)
        desc = featureVideo.next()
        if not self.sampler:
            self.sampler = Sampler(self.featureController.height, self.featureController.width, self.step)
        # Sampling the descriptor
        descSampled = self.sampler.sampleDescriptor(desc, 3)

        # CLASSIFING
        if self.step != 1:
            resultSampled = self.classifier.predict(descSampled)
            result[self.sampler.indexes] = resultSampled
        else:
            result = self.classifier.predict(descSampled)

        # RESHAPING
        result = result.reshape((self.featureController.height,
                        self.featureController.width,1))
        result = Frame(result)
        return result

