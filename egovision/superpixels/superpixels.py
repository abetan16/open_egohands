__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['Superpixels']


import sys, os
sys.path.append(os.path.abspath('../../'))
from _slic import _slic_cython
import numpy as np
from egovision.performance import getConfusionMatrix
from egovision.handSegmentation.postProcessor import ProbabilityThresholder
from egovision.handSegmentation import SegmenterEvaluator
from egovision import Frame
from egovision.interfaces import cv2i
from skimage.segmentation import mark_boundaries
from utils import regular_grid
import time
import copy
from egovision.output.visualizer import VideoVisualizer


class Superpixels():

    def __init__(self, nSegments, compactness, compressionWidth, sigma):
        self.nSegments = nSegments
        self.compactness = compactness
        self.initialSolution = None
        self.compressionWidth = compressionWidth
        self.method = "slic"
        self.lastSegments = None
        self.sigma = sigma
        self.probabilityThresholder = ProbabilityThresholder(self.compressionWidth, 0.4)

    def slic(self, frame):
        if frame.matrix.shape[1] == self.compressionWidth:
            frameCompressed = frame
            compressed = False
        else:
            frameCompressed = frame.resizeByWidth(self.compressionWidth)
            compressed = True
        frameCompressed.matrix = cv2i.GaussianBlur(frameCompressed.matrix,self.sigma,0,0,cv2i.BORDER_REFLECT)
        # Initialize parameters
        image = frameCompressed.matrix
        n_segments = self.nSegments
        compactness = self.compactness
        segments = self.lastSegments
        sigma = 0 
        max_iter = 10
        spacing = np.ones(3)
        slic_zero = True
        
        image = image[np.newaxis, ...]

        depth, height, width = image.shape[:3]

        # initialize cluster centroids for desired number of segments
        grid_z, grid_y, grid_x = np.mgrid[:depth, :height, :width]

        slices = regular_grid(image.shape[:3], n_segments)
        step_z, step_y, step_x = [int(s.step) for s in slices]
        if segments is None:
            segments_z = grid_z[slices]
            segments_y = grid_y[slices]
            segments_x = grid_x[slices]

            segments_color = np.zeros(segments_z.shape + (image.shape[3],))
            segments = np.concatenate([segments_z[..., np.newaxis],
                                       segments_y[..., np.newaxis],
                                       segments_x[..., np.newaxis],
                                       segments_color],
                                      axis=-1).reshape(-1, 3 + image.shape[3])
            segments = np.ascontiguousarray(segments)


        # we do the scaling of ratio in the same way as in the SLIC paper
        # so the values have the same meaning
        step = float(max((step_z, step_y, step_x)))
        ratio = 1.0 / compactness
        
        image = np.ascontiguousarray(image * ratio)
        
        t0 = time.time()
        labels, segments = _slic_cython(image, segments, step, max_iter, spacing, slic_zero)
        labels = np.array(labels)
        segments = np.array(segments)
        means = copy.deepcopy(segments)
        t1 = time.time()

        labels = labels[0]
        means[:,3:] = means[:,3:]/ratio

        self.executionTime = t1-t0
        self.lastMeans = means
        self.lastNumberOfPixels = len(means)
        self.lastSegments = segments
        self.lastSuperpixelId = np.max(labels)
        result = Frame(labels.astype(np.uint8))
        result.matrix = result.matrix.reshape((result.matrix.shape[0],
                                           result.matrix.shape[1],
                                           1))
        self.superpixels2pixels = {}
        for spxIndex in range(self.lastSuperpixelId):
            pixels = np.where(labels==spxIndex)
            self.superpixels2pixels[spxIndex] = pixels

        self.lastLabels = result
        self.lastFrame = frameCompressed

        return result

    def train(self, handSegmenter, dataManagerList):
        def superpixels2pixelPredictions(superpixels, predictions):
            maxCategory = np.max(superpixels.matrix)
            maskFloat = np.zeros((superpixels.matrix.shape[0],superpixels.matrix.shape[1],1))
            for category in range(maxCategory):
                maskFloat[superpixel.superpixels2pixels[category]] = predictions[category]
            result = Frame(maskFloat)
            return result

        th = 0.4
        optimalSegments = 0
        optimalCompactness = 0
        optimalObjFunction = 0
        for nSegments in xrange(50,450,50):
            for compactness in xrange(4,20,3):
                self.nSegments = nSegments
                self.compactness = compactness
                self.lastSegments = None
                of = 0
                
                for ns, dataManager in enumerate(dataManagerList):
                    superpixel = Superpixels(nSegments,
                                           compactness,
                                           self.compressionWidth, 
                                           self.sigma)
                    #LOADING THE MASK
                    frame = Frame.fromFile(dataManager.frameFiles[0],compressionWidth=self.compressionWidth)
                    frame = frame.fromBGR2ColorSpace(handSegmenter.feature)
                    mask = Frame.loadMask(dataManager.maskFiles[0], self.compressionWidth)
                    slicFrame = superpixel.slic(frame)

                    nans = np.isnan(superpixel.lastMeans)
                    if np.any(nans):
                        superpixel.lastMeans[nans] = -1
                    handProbabilities = handSegmenter.__getClassifier__(ns).predict(superpixel.lastMeans[:,3:]) 
                    handProbabilities = superpixels2pixelPredictions(superpixel.lastLabels, handProbabilities)
                    contours = self.probabilityThresholder.process(handProbabilities)
                    segment = Frame(self.probabilityThresholder.__getFrameMatrix__(contours, mask))
                    segment.matrix = segment.matrix/255

                    cm, f0, f1 = SegmenterEvaluator.evaluate(mask, segment)

                    of = of + f1

                if of > optimalObjFunction:
                    optimalObjFunction = of
                    optimalSegments = nSegments
                    optimalCompactness = compactness
                    print optimalSegments, optimalCompactness, optimalObjFunction
        
        self.nSegments = optimalSegments
        self.compactness = optimalCompactness
        self.lastMeans = superpixel.lastMeans
        print self.nSegments, self.compactness, optimalObjFunction

    def boundaryMask(self, segments):
        segments = segments.matrix.reshape((self.lastFrame.matrix.shape[0],self.lastFrame.matrix.shape[1]))
        whiteMask = np.zeros_like(segments)
        result = mark_boundaries(whiteMask, segments,color=(1,1,1),mode='thick')*255
        result = Frame(result.astype(np.uint8))
        return result

