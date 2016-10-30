from superpixels import Superpixels
import numpy as np
import math
import time
import sys, os
sys.path.append(os.path.abspath('../../'))
from egovision import Frame
from egovision.performance import getConfusionMatrix
from egovision.performance import getFScore
from sklearn.ensemble import RandomForestRegressor


class MultiModelSuperpixels(Superpixels):
    
    def __init__(self,nModels,compressionWidth):
        self.nModels = nModels
        self.compressionWidth = compressionWidth
        self.models = [None for x in range(nModels)]

    def train(self, multiModelHandSegmenter, handSegmenterDataManager):
        for nm in range(self.nModels):
            model = self.__optimizeSuperpixels__(multiModelHandSegmenter.models[nm],
                                         handSegmenterDataManager[nm],
                                         xrange(50,250,50),
                                         xrange(3,15,3))
            print model.nSegments, model.compactness
            self.models[nm] = model
        return model
    


    def __optimizeSuperpixels__(self,handSegmenter, dataManager, nSegmentsChoices, compactnessChoices,probabilityMap=None):
        th = 0.4
        frame = Frame.fromFile(dataManager.frameFiles[0])
        mask = Frame.loadMask(dataManager.maskFiles[0])
        mask = mask.resizeByWidth(self.compressionWidth)
        groundTruth = mask.matrix.flatten()
        optimalScore = 0
        result = None
        frame2 = frame.fromBGR2ColorSpace("LAB")
        frame2 = frame2.resizeByWidth(self.compressionWidth)
        desc = frame2.matrix.reshape(frame2.matrix.shape[0]*frame2.matrix.shape[1],3)
        handSegmenter.classifier.fit(desc,groundTruth)
        for nSegments in nSegmentsChoices:
            for compactness in compactnessChoices:
                superpixel = Superpixels(nSegments, compactness,self.compressionWidth)
                segmentation = handSegmenter.segmentFrame(frame, "slic",
                                    superpixel=superpixel,
                                    probabilityMap=probabilityMap) 
                segmentation.matrix[segmentation.matrix>=th] = 1
                segmentation.matrix[segmentation.matrix<th] = 0
                prediction = segmentation.matrix.flatten()
                confusion = getConfusionMatrix(groundTruth, prediction)
                of = confusion[0][0][0] + confusion[0][1][1]
                #f0, f1 = getFScore(confusion[0])
                print nSegments, compactness, of, np.max(superpixel.lastLabels.matrix)
                if of > optimalScore:
                    optimalScore = of
                    result = superpixel
        return result
        
        

