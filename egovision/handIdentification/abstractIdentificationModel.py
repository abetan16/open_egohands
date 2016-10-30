"""

Assuming non-occluded hand-segments, the next step is to decide if they are
left or right.  This figure shows the objective of the hand-identification
level. A quick analysis of egocentric videos of daily activities easily points
to the angle of the hands with respect to the lower frame border
(:math:`\\theta`), and the normalized horizontal distance to the left border
(:math:`x`) as two discriminative variables to build our L/R
hand-identification model. 

.. image:: ../_images/handIdentification/geometricProblem.png
    :align: center

In general our design contains an abstractIdentificationModel that contains
common functionalities such as identify a contour and visualize the probability
function. The role of the inherited classes is to define the left and right
probability functions via masks or by defining its parameters. 

.. image:: ../_images/handIdentification/identificationStructure.png
    :align: center

"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['HandSegmenter']

from abc import ABCMeta, abstractmethod
import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import numpy as np
from egovision.interfaces import cv2i
from egovision.output.utils import cvEllipse2points

class AbstractIdentificationModel(object):
    __metaclass__ = ABCMeta
    
    def __init__(self, compressionWidth):
        self.compressionWidth = compressionWidth
        self.leftFunction = lambda x: 0
        self.rigthFunction = lambda y: 0


    @abstractmethod
    def fit(self, *args, **kargs):
        pass
    
    def identifyContours(self, contours):
        """

        Once the identification model is defined the next step identify
        segmented contours as left or right hand. This can be done by a
        this likelihood ratio test:

        .. math::
           :nowrap:

            \\begin{equation}
            \\Lambda(x, \\theta) = \\dfrac{L_{l}(\\Theta_l^x,\\Theta_l^{\\theta}|x, \\theta)} {L_{r} (\\Theta_r^x,\\Theta_r^{\\theta}|x,\\theta)} =  \\dfrac{p_{l}(x, \\theta|\\Theta_l^x,\\Theta_l^{\\theta})} {p_{r} (x,\\theta|\Theta_r^x,\\Theta_r^{\\theta})}
            \end{equation}
        
        Relying only on the likelihood ratio, could lead to cases where two
        hand-like segments are assigned the same label (left or right). To
        avoid this cases, and given that a frame cannot have two left nor two
        right hands, we follow a competitive rule in the following way. Lets
        assume two hands-like segments in the frame described by :math:`z_1 =
        (x_1,\\theta_1)` and :math:`z_2 = (x_2,\\theta_2)`, and their respective
        likelihood ratios given by :math:`\\Lambda(x_1,\\theta_1)` and
        :math:`\Lambda(x_2,\\theta_2)`. The competitive ids are assigned in the
        following way:
 

        .. math::
           :nowrap:

            \\begin{eqnarray}
            id_{z_1}, id_{z_2} &=& \\begin{cases}
                    \\Lambda(x_1,\\theta_1) > \\Lambda(x_2,\\theta_2) \\rightarrow & id_{z_1} = l \\\\
                                           & id_{z_2} = r \\\\ \\\\
                    \\Lambda(x_1,\\theta_1) \\le \\Lambda(x_2,\\theta_2) \\rightarrow & id_{z_1} = r \\\\
                                           & id_{z_2} = l
                    \\end{cases}
           \\end{eqnarray}

        
        This image is the result of the following example.

        .. image:: ../_images/handIdentification/handIdentificationExample.png
            :align: center

        Example 1: Visualizing the hand-identification::

            from egovision.values.colors import HAND_COLOR
            from egovision.values.paths import DATASET_MASKS_PATH
            from egovision.values.paths import DATASET_FRAMES_PATH
            from egovision import Frame
            from egovision.handSegmentation import PixelByPixelHandSegmenter
            from egovision.handSegmentation import PostB2016
            from egovision.handIdentification import MaxwellIdentificationModel
            from egovision.handSegmentation import SegmentVisualizer
            
            # DEFINING SOME PARAMETERS    
            COMPRESSION_WIDTH = 400
            CLASSIFIER = "RF"
            STEP = 2
            FEATURE = "LAB"
            DATASET = "GTEA"
            TRAININGMASK = "00000180.jpg"
            TRAININGVIDEO = "GTEA_S1_Coffee_C1"
            POSTPROCESS_PARAMETERS = {
                "sigma" : (9, 9),
                "probabilityThreshold" : 0.2,
                "marginPercentage" : 0.02,
                "compressionWidth" : COMPRESSION_WIDTH,
                "minimumAreaPercentage" : 0.005,
                "maxContours" : 3 }
            postProcessor = PostB2016(**POSTPROCESS_PARAMETERS)
            
            
            # DEFININF THE IDENTIFICATION MODEL
            ID_PARAMS = [[-0.05357189,0.23923865,-0.63053494,0.94603864], \\
                         [-0.0851523, 0.21251223, -0.91460727,1.10119661]]
            idModel = MaxwellIdentificationModel(COMPRESSION_WIDTH)
            idModel.setParameters(ID_PARAMS)
            
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
            
            # IDENTIFYING
            contours, identities = idModel.identifyContours(postProcessor.others["contours"])
            
            # VISUALIZE
            for nc, contour in enumerate(contours):
                frame = SegmentVisualizer.__overlapContour__(frame, segment,
                                                             contour,
                                                             HAND_COLOR[identities[nc]])
            
            SegmentVisualizer.showFrameQuick("IDENTIFICATION", frame) 

        """
        handsIds = {0:0, 1:0}
        ellipses = []
        if contours:
            contours, ellipses, probabilities = self.__contours2probabilities__(contours)
            maxIndex = np.argmax(probabilities)
            winner = maxIndex/2
            winnerId = maxIndex%2
            looser = (winner + 1)%2
            looserId = (winnerId + 1)%2
            handsIds[winner] = winnerId
            handsIds[looser] = looserId
        return contours, ellipses, handsIds
    
    def contours2ellipses(self, contours):
        return self.__contours2ellipses__(contours)
    
    @classmethod
    def __contours2ellipses__(cls, contours):
        contours.sort(key=lambda x: cv2i.contourArea(x),reverse=True)
        # contours = contours[:2]
        return contours, np.array(map(cv2i.fitEllipse, contours))

    def __ellipse2probabilities__(self, ellipses):
        centroids = []
        for x in ellipses:
            center, a11, a12, a21, a22 = cvEllipse2points(x)
            if center[0] > 0:
                centroids.append(center[0]/float(self.compressionWidth))
            else:
                centroids.append(0.0)
        centroids = np.array(centroids)
        angles = np.array(map(lambda x: np.radians(x[2]), ellipses))
        coords_left = zip(centroids, angles)
        coords_right = zip(1 - centroids, np.pi - angles)
        leftProbabilities = map(lambda x: self.leftFunction(x), coords_left)
        rightProbabilities = map(lambda x: self.rightFunction(x), coords_right)
        probabilities = np.vstack([leftProbabilities, rightProbabilities]).T
        return probabilities
        

    def __contours2probabilities__(self, contours):
        contours, ellipses = self.contours2ellipses(contours)
        probabilities = self.__ellipse2probabilities__(ellipses)
        return contours, ellipses, probabilities

    def __getIdentificationGrid__(self):
        angleStep = np.pi/300.0
        sizeStep = 0.01
        angles, x = np.mgrid[slice(0,np.pi + angleStep, angleStep),
                           slice(0,1 + sizeStep, sizeStep)]
        leftGrid = np.zeros_like(angles).astype(float)
        rightGrid = np.zeros_like(angles).astype(float)
        for na in range(len(angles)):
            for nr in range(len(angles[0])):
                coord_left = x[na,nr],angles[na,nr]
                leftGrid[na,nr] = self.leftFunction(coord_left)
                coord_right = x[na, nr], np.pi-angles[na, nr]
                rightGrid[na,nr] = self.rightFunction(coord_right)
        return angles, x, leftGrid, rightGrid

    def __visualizeHandDistribution__(self, angles, x, grid, subplot, polar=False):
        if polar:
            subplot.pcolor(angles, x, grid)   
            subplot.axis([0,np.pi,0,1])
        else:
            subplot.pcolor(x, angles, grid)   
            subplot.axis([0, 1, 0, np.pi])
        

    def visualize(self, polar=True):
        angles, x, leftGrid, rightGrid = self.__getIdentificationGrid__()
        fig = plt.figure()
        plt.subplot(221,polar=polar)
        plt.title("Left Hand")
        self.__visualizeHand__(angles, x, leftGrid, polar)
        plt.colorbar()
        # data = np.array(self.leftMeasurements).T
        # plt.scatter(data[1],data[0])
        plt.subplot(222,polar=polar)
        self.__visualizeHand__(angles, x, rightGrid, polar)
        plt.title("Right Hand")
        plt.colorbar()
        # data = np.array(self.rightMeasurements).T
        # plt.scatter(np.pi - data[1],data[0])
        plt.show()

