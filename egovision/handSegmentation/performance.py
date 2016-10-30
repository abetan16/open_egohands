__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['evaluateSegmentation','evaluateLeftRightSegmentation']

import sys, os
sys.path.append(os.path.abspath('../'))
from egovision.performance import getConfusionMatrix
from egovision.performance import getFScore

class SegmenterEvaluator:
    """ 

    A good way to evaluate the segmentation performance is using the confusion
    matrix and the F0 and F1 score. The literature frequently use the f1 score
    to compare different hand segmenters. The following example takes the
    manual mask as ground truth.

    
    Example 1: Evaluating the hand-segmenter::
    
        from egovision.values.paths import DATASET_MASKS_PATH
        from egovision.values.paths import DATASET_FRAMES_PATH
        from egovision import Frame
        from egovision.handSegmentation import PixelByPixelHandSegmenter
        from egovision.handSegmentation import PostB2016
        from egovision.handSegmentation import SegmenterEvaluator
        
        
        # DEFINING SOME PARAMETERS    
        COMPRESSION_WIDTH = 200
        CLASSIFIER = "RF"
        STEP = 2
        FEATURE = "LAB"
        DATASET = "GTEA"
        TRAININGMASK = "00000360.jpg"
        TRAININGVIDEO = "GTEA_S1_Coffee_C1"
        TESTINGVIDEO = "GTEA_S1_Tea_C1"
        POSTPROCESS_PARAMETERS = {
            "sigma" : (9, 9),
            "probabilityThreshold" : 0.2,
            "marginPercentage" : 0.02,
            "compressionWidth" : COMPRESSION_WIDTH,
            "minimumAreaPercentage" : 0.005,
            "maxContours" : 3 }
        postProcessor = PostB2016(**POSTPROCESS_PARAMETERS)
        
        
        # LOADING THE TRAINING MASK AND FRAME            
        mask = DATASET_MASKS_PATH.format(DATASET, TRAININGVIDEO) + TRAININGMASK
        frameFile = DATASET_FRAMES_PATH.format(DATASET, TRAININGVIDEO) + TRAININGMASK
        trainingframe = Frame.fromFile(frameFile)
        
        
        # LOADING THE TESTING MASK AND FRAME
        testingMask = DATASET_MASKS_PATH.format(DATASET, TESTINGVIDEO) + TRAININGMASK
        testingMask = Frame.loadMask(testingMask, compressionWidth = COMPRESSION_WIDTH)
        frameFile = DATASET_FRAMES_PATH.format(DATASET, TESTINGVIDEO) + TRAININGMASK
        testingFrame = Frame.fromFile(frameFile)
        
        
        # TRAINING THE HAND-SEGMENTER 
        hs = PixelByPixelHandSegmenter(FEATURE, COMPRESSION_WIDTH, CLASSIFIER, STEP)
        dataManager = hs.trainClassifier([mask])
        
        
        # SEGMENTING THE FRAME
        segment = hs.segmentFrame(testingFrame)
        segment = postProcessor.process(segment)
        postProcessor.visualize(segment)
        
        # EVALUATING THE SEGMENTER
        cm, f0, f1 = SegmenterEvaluator.evaluate(testingMask, segment)
        
        print cm, f0, f1


    Example 2: Training Coffe_20_10, testing on Tea ::

        from egovision.values.paths import DATASET_MASKS_PATH
        from egovision.values.paths import DATASET_FRAMES_PATH
        from egovision.values.paths import DATASET_VIDEOS_PATH
        from egovision import Frame
        from egovision.handSegmentation import PixelByPixelMultiHandSegmenter
        from egovision.handSegmentation import PostB2016
        from egovision.handSegmentation import SegmenterEvaluator
        from egovision.modelRecommenders import NearestNeighbors
        import numpy as np
        import os
        
        # DEFINING SOME PARAMETERS    
        N_MODELS = 20
        K_AVERAGE = 10
        GLOBAL_FEATURE = "HSV-HIST"
        COMPRESSION_WIDTH = 200
        CLASSIFIER = "RF"
        STEP = 2
        FEATURE = "LAB"
        DATASET = "GTEA"
        TRAININGMASK = "00000360.jpg"
        TRAININGVIDEO = "GTEA_S1_Coffee_C1"
        TESTINGVIDEO = "GTEA_S1_Tea_C1"
        TESTINGMASK = "00006060.jpg"
        POSTPROCESS_PARAMETERS = {
            "sigma" : (3, 3),
            "probabilityThreshold" : 0.2,
            "marginPercentage" : 0.1,
            "compressionWidth" : COMPRESSION_WIDTH,
            "minimumAreaPercentage" : 0.005,
            "maxContours" : 3 }
        postProcessor = PostB2016(**POSTPROCESS_PARAMETERS)
        
        
        # LOADING THE TRAINING MASK AND FRAME            
        mask = DATASET_MASKS_PATH.format(DATASET, TRAININGVIDEO)
        frameFile = DATASET_FRAMES_PATH.format(DATASET, TRAININGVIDEO)
        allMasks = [mask + x for x in os.listdir(mask)]
        allFrames = [frameFile + x for x in os.listdir(mask)]
        
        
        # TRAINING THE MODEL RECOMMENDER
        modelRecommender = NearestNeighbors(K_AVERAGE, COMPRESSION_WIDTH, GLOBAL_FEATURE)
        modelRecommender.train(allFrames[:N_MODELS])
        
        
        # TRAINING THE MULTIMODEL PIXEL BY PIXEL
        hs = PixelByPixelMultiHandSegmenter(FEATURE, COMPRESSION_WIDTH, CLASSIFIER, STEP)
        hs.trainClassifier(allMasks[:N_MODELS], modelRecommender)
        
        
        # SEGMENTING THE FRAME
        mask = DATASET_MASKS_PATH.format(DATASET, TESTINGVIDEO)
        frameFile = DATASET_FRAMES_PATH.format(DATASET, TESTINGVIDEO)
        allMasks = [mask + x for x in os.listdir(mask)]
        allFrames = [frameFile + x for x in os.listdir(mask)]
        cm = np.zeros((2,2))
        
        
        # PER MASK
        for nf, maskName in enumerate(allMasks):
            frame = Frame.fromFile(allFrames[nf])
            testingMask = Frame.loadMask(allMasks[nf], compressionWidth=COMPRESSION_WIDTH)
            segment = hs.segmentFrame(frame)
            segment = postProcessor.process(segment)
            postProcessor.visualize(segment)
            cmi = SegmenterEvaluator.getConfusionMatrix(testingMask, segment)
            cm += cmi
        
        # TOTAL F1
        F0, F1 = SegmenterEvaluator.getFScore(cm)
        print cm, F0, F1 


    """

    @classmethod
    def evaluate(cls, mask, segment):
        prediction = segment.matrix.flatten()
        groundTruth = mask.matrix.flatten()
        cmi = getConfusionMatrix(groundTruth, prediction, 2)
        f0, f1 = getFScore(cmi[0])
        return cmi[0], f0, f1

    @classmethod
    def getConfusionMatrix(cls, mask, segment):
        prediction = segment.matrix.flatten()
        groundTruth = mask.matrix.flatten()
        cmi = getConfusionMatrix(groundTruth, prediction, 2)
        return cmi[0]

    @classmethod
    def getFScore(cls, cm):
        f0, f1 = getFScore(cm)
        return f0, f1
