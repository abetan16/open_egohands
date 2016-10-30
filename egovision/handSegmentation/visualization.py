__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['VideoVisualizer']

from egovision.interfaces import cv2i
from egovision.output.visualizer import VideoVisualizer
from egovision.output.utils import cvEllipse2points
from egovision.values.colors import CENTER_COLOR, MINOR_AXE_COLOR, MAYOR_AXE_COLOR
import numpy as np
from egovision import Frame
from utils import scaleEllipse




class SegmentVisualizer(VideoVisualizer):
    """
    
    A perfect way to understand the performance of the hand segmenter is by
    visualizing it as a video sequence. The next example show how to segment,
    postrpocess, and overlap the result on the original videosequence. The
    following video shows the result.

    .. youtube:: https://www.youtube.com/watch?v=jmkayWZA5A0

    * Example 1: Segmenting the Tea_S1 video sequence::

        from egovision.values.paths import DATASET_MASKS_PATH
        from egovision.values.paths import DATASET_FRAMES_PATH
        from egovision.values.paths import DATASET_VIDEOS_PATH
        from egovision import Frame, Video
        from egovision.handSegmentation import PixelByPixelMultiHandSegmenter
        from egovision.handSegmentation import PostB2016
        from egovision.handSegmentation import SegmentVisualizer
        from egovision.modelRecommenders import NearestNeighbors
        import numpy as np
        import os
        
        # DEFINING SOME PARAMETERS    
        N_MODELS = 20
        K_AVERAGE = 10
        GLOBAL_FEATURE = "HSV-HIST"
        COMPRESSION_WIDTH = 400
        CLASSIFIER = "RF"
        STEP = 2
        FEATURE = "LAB"
        DATASET = "GTEA"
        TRAININGMASK = "00000360.jpg"
        TRAININGVIDEO = "GTEA_S1_Coffee_C1"
        TESTINGVIDEO = "GTEA_S1_Tea_C1"
        TESTINGMASK = "00006060.jpg"
        POSTPROCESS_PARAMETERS = {
            "sigma" : (9, 9),
            "probabilityThreshold" : 0.2,
            "marginPercentage" : 0.01,
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
        
        
        # LOADING TESTING VIDEO
        video = Video(DATASET_VIDEOS_PATH.format(DATASET, TESTINGVIDEO) + ".MP4")
        
        # PER MASK
        for frame in video:
            binaryShape = (frame.matrix.shape[0], frame.matrix.shape[1], 1)
            segment = hs.segmentFrame(frame)
            segment = postProcessor.process(segment, targetShape = binaryShape)
            frame = SegmentVisualizer.__overlapSegmentation__(frame, segment, (0,0,255))
            SegmentVisualizer.showFrameQuick("BENCHTEST", frame)


    """
    

    @classmethod
    def __overlapContour__(self, frame, segment, contour, color):
        """


        """
        segment = cv2i.contours2binary([contour], segment.matrix.shape)
        return self.__overlapSegmentation__(frame, Frame(segment), color)

    @classmethod
    def __overlapSegmentation__(self, frame, segmentation, color):
        """


        """
        height = frame.matrix.shape[0]
        width = frame.matrix.shape[1]
        segmentationResized = segmentation.resize(height, width)
        mask = np.ones(frame.matrix.shape, np.uint8)
        if frame.matrix.shape[-1] == 3:
            mask = (mask*segmentationResized.matrix*color).astype("uint8")
        else:
            mask = (mask*segmentationResized.matrix*[255]).astype("uint8")

        matrix = cv2i.addWeighted(frame.matrix, 1, mask, 0.4, 0)
        result = Frame(matrix)
        return result

        

    @classmethod
    def __overlapEllipse__(self, frame, segment, ellipse, color):
        ellipseScaled = scaleEllipse(frame, segment, ellipse)
        cv2i.ellipse(frame.matrix,ellipseScaled,color,2,1)        
        center, a11, a12, a21, a22 = cvEllipse2points(ellipseScaled)
        pointSize = 5
        cv2i.circle(frame.matrix, tuple(map(int,center)), 5, CENTER_COLOR, -1, 8);
        cv2i.circle(frame.matrix, tuple(map(int,a11)), 5, MINOR_AXE_COLOR, -1, 8);
        cv2i.circle(frame.matrix, tuple(map(int,a12)), 5, MINOR_AXE_COLOR, -1, 8);
        cv2i.circle(frame.matrix, tuple(map(int,a21)), 5, MAYOR_AXE_COLOR, -1, 8);
        cv2i.circle(frame.matrix, tuple(map(int,a22)), 5, MAYOR_AXE_COLOR, -1, 8);
        return frame
    
    @classmethod
    def __overlapBinaryEllipse__(self, frame, segment, ellipse):
        ellipseScaled = scaleEllipse(frame, segment, ellipse)
        cv2i.ellipse(frame.matrix,ellipseScaled,(255),-1, 1)        
        return frame
