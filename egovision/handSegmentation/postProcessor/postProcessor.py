""" 

In general we design the hand-segmentater postProcessing as an object that
recieves a binary frame or contour and returns a filtered version of it. The
type of post processing steps are not restricted the library design. The
proposed design is based on AbstractPostProcessor: An object that defines the
input and output types and implement comon functionalities such as process()
and visualize(). 

.. image:: ../_images/diagrams/postProcessor.png
    :align: center

These are some examples of postProcessors implemented in EgoHands.

    Mask <- ProbabilitySmoother(Mask): Return a blured version of a binary mask.

    Contout <- ProbabilityThresholder(Mask): Find the contours containing the
    most probable areas
    
    Contour <- AreaFilter(Contour): Keeps only contours larger than a
    particular areaThreshol.
    
    Contour <- BorderFilter(Contour): Keeps only contours close to the left,
    right and lower border.
    
    Contour <- NumberFilter(Contour): Keep the N largest contours.

It is easy to extend the funcionality or create new postProcessing steps. As an
example we have developed a complex PostProcessor [Bet2016] which chains the
steps above and redefine the visualize method.

"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"

from numberFilter import NumberFilter
from areaFilter import AreaFilter
from borderFilter import BorderFilter
from contourThresholder import ProbabilityThresholder
from probabilitySmoother import ProbabilitySmoother
from abstractPostProcessor import AbstractPostProcessor
from egovision import Frame
import numpy as np
from egovision.output import VideoVisualizer
from egovision.interfaces import cv2i

class PostB2016(AbstractPostProcessor):
    """ 
    There are some intuitive decisions that can help to reduce the
    number of false positives of the hand segmenters. This method apply
    these rules in the resulting probability map and returns the valid
    contours and discarted contours. In our more recent paper we found that
    these rules could help to improve the segmenter around 10%.:

    #. Blur the probability map to obtain a smoothed version and reduce the \
    impact of the sampling used by the hand-segmenter.
    #. Remove the contours that are far of the left, bottom and right border\
    #. Remove small contours
    #. Keep the larger 3 contours of the valid contours

    The following image shows an example of the postprocessing steps. This
    image is generated in the example bellow:

    :ivar Frame frame: probability map comming from the hand segmenter.
    
    :returns: defined output of the class

    .. image:: ../_images/handSegmentation/postProcessingSteps.png
        :align: center

    Example 1: Using and post-processing the hand-segmenter results::

        from egovision.values.paths import DATASET_MASKS_PATH
        from egovision.values.paths import DATASET_FRAMES_PATH
        from egovision import Frame
        from egovision.handSegmentation import PixelByPixelHandSegmenter
        from egovision.handSegmentation import PostB2016


        # DEFINING SOME PARAMETERS    
        COMPRESSION_WIDTH = 200
        CLASSIFIER = "RF"
        STEP = 2
        FEATURE = "LAB"
        DATASET = "GTEA"
        TRAININGMASK = "00000360.jpg"
        TRAININGVIDEO = "GTEA_S1_Coffee_C1"
        POSTPROCESS_PARAMETERS = {
            "sigma" : (9, 9),
            "probabilityThreshold" : 0.2,
            "marginPercentage" : 0.02,
            "compressionWidth" : COMPRESSION_WIDTH,
            "minimumAreaPercentage" : 0.005,
            "maxContours" : 3 }
        postProcessor = PostB2016(**POSTPROCESS_PARAMETERS)


        # DEFINING THE MASK AND FRAME FILE FOR THIS EXAMPLE            
        mask = DATASET_MASKS_PATH.format(DATASET, TRAININGVIDEO) + TRAININGMASK
        frameFile = DATASET_FRAMES_PATH.format(DATASET, TRAININGVIDEO) + TRAININGMASK
        frame = Frame.fromFile(frameFile)


        # TRAINING THE HAND-SEGMENTER 
        hs = PixelByPixelHandSegmenter(FEATURE, COMPRESSION_WIDTH, CLASSIFIER, STEP)
        dataManager = hs.trainClassifier([mask])


        # SEGMENTING THE FRAME
        segment = hs.segmentFrame(frame)


        # POSTPROCESSING THE RESULTS
        contours = postProcessor.process(segment)
        segmentedMask = postProcessor.__getOutputFrame__(segment)
        postProcessor.visualize(segment)

    """

    def __init__(self, sigma = (3,3),
                       probabilityThreshold = 0.4,
                       marginPercentage = 0.05,
                       compressionWidth = 200,
                       minimumAreaPercentage = 0.01,
                       maxContours = 3):
        AbstractPostProcessor.__init__(self, Frame, Frame, compressionWidth)
        self.smoother = ProbabilitySmoother(compressionWidth, sigma)
        self.thresholder = ProbabilityThresholder(compressionWidth, probabilityThreshold)
        self.borderFilter = BorderFilter(compressionWidth, marginPercentage)
        self.areaFilter = AreaFilter(compressionWidth, minimumAreaPercentage)
        self.numberFilter = NumberFilter(compressionWidth, maxContours)
        self.output = None
        self.others = {}


    def process(self, frame, targetShape=None):
        self.input = frame
        smoothedMap = self.smoother.process(frame) # Frame -> Frame
        contours = self.thresholder.process(smoothedMap) # Frame -> Contours
        contours = self.borderFilter.process(contours, frame.matrix.shape) # Contours -> Contours
        contours = self.areaFilter.process(contours, frame.matrix.shape) # Contours -> Contours
        contours = self.numberFilter.process(contours) # Contours -> Contours
        self.output = contours
        segment = self.__getOutputFrame__(frame, targetShape)
        segment.matrix /= 255
        self.output = segment
        self.others["smoothedMap"] = smoothedMap 
        self.others["contours"] = contours
        return segment

    def visualize(self, frame):
        visualizer = VideoVisualizer()
        steps = []
        inputProbability = self.__getFrameMatrix__(self.input, frame)
        steps.append(inputProbability)
        steps.append(self.smoother.__getFrameMatrix__(self.smoother.output, frame))
        steps.append(self.thresholder.__getFrameMatrix__(self.thresholder.output, frame))
        steps.append(self.borderFilter.__getFrameMatrix__(self.borderFilter.output, frame))
        steps.append(self.areaFilter.__getFrameMatrix__(self.areaFilter.output, frame))
        steps.append(self.numberFilter.__getFrameMatrix__(self.numberFilter.output, frame))
        visualizeFrame = Frame(np.hstack(steps))
        visualizer.showFrameQuick(type(self).__name__, visualizeFrame)
        for i in range(100):
            cv2i.waitKey(1)
