""" 
    

"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['IdentificationModel']

from egovision import Frame
from egovision.handSegmentation.postProcessor import ProbabilityThresholder
from abstractIdentificationModel import AbstractIdentificationModel
from egovision.interfaces import cv2i
import numpy as np
import os


class KernelIdentificationModel(AbstractIdentificationModel):
    """
    
    It is possible to obtain a identification model using left and right masks.
    The main idea is to find the probability of a countour given its horizonal
    position (:math:`x`) and its angle (:math:`\\theta`). In our unpublished
    work [CITE] the kitchen dataset is extended with the left and right masks
    as the following figure shows.

    .. image:: ../_images/handIdentification/leftRightMask.png
        :align: center

    In egohands the filename of each left/right mask is the number of the frame
    filled by zeros followed by "_0" and "_1" for left and right mask
    respectively. This is the final structure of the path::
        
        # LEFT MASK OF FRAME 180
        <base_path>/<dataset>/LRmasks/<VideoName>/00000180_0.jpg

        # GTEA EXAMPLE
        egovision/dataExamples/GTEA/LRmasks/GTEA_S1_Coffee_C1/00000180_0.jpg

        # RIGHT MASK OF FRAME 501
        <base_path>/<dataset>/LRmasks/<VideoName>/00000180_1.jpg

        # GTEA EXAMPLE
        egovision/dataExamples/GTEA/LRmasks/GTEA_S1_Coffee_C1/00000180_0.jpg

    """

    def __init__(self, compressionWidth = 200, threshold = 0.8):
        AbstractIdentificationModel.__init__(self, compressionWidth)
        self.threshold = threshold
        self.probabilityThresholder = ProbabilityThresholder(compressionWidth, threshold)
        self.leftMeasurements = []
        self.rightMeasurements = []

    def fit(self, maskFolders):
        """

        Once collected the data and saved in the appropiate format. The next
        step is to estimate the probability surface covering the space
        :math:`x\\times\\alpha`. We accomplish this by fitting an ellipse to
        the mask contours and estimating the probability kernel. This figure
        shows the probability distribution of the left and the right hand
        respectively.  

        
        .. image:: ../_images/handIdentification/KernelModel.png
            :align: center

        In  the  horizontal  axis is the relative distance to the left border
        (:math:`x`), for the left hand-like segments, and the relative distance
        to the right border (:math:`1-x`), for the right hand-like segments.
        The secondary axis is the angle of the fitted ellipse with respect to
        the border of the frame (:math:`\\theta`). The following example shows
        the fiting and visualization process.

        Example 1: Empirical Hand Identifier::

            from egovision.handIdentification import MaxwellIdentificationModel
            from egovision.handIdentification import KernelIdentificationModel
            from egovision.values.paths import DATASET_LRMASKS_PATH
            
            DATASET = "GTEA"
            VIDEO_NAMES = ["GTEA_S1_Coffee_C1", "GTEA_S1_CofHoney_C1", \\
                           "GTEA_S1_Hotdog_C1", "GTEA_S1_Pealate_C1",\\
                           "GTEA_S1_Peanut_C1", "GTEA_S1_Tea_C1"] 
            VIDEO_FILES = [DATASET_LRMASKS_PATH.format(DATASET, x) for x in VIDEO_NAMES]
            
            idModel = KernelIdentificationModel()
            idModel.fit(VIDEO_FILES)
            idModel.visualize()

        """
        map(self.__loadLeftRightMask__, maskFolders)
        self.__setIdentificationFunctions__()

    def __loadLeftRightMask__(self, maskFolder):
        pointsLeft = []
        pointsRight = []
        maskFiles = os.listdir(maskFolder)
        for nm, mask in enumerate(maskFiles):
            maskFrame = Frame.loadMask(maskFolder + mask, self.compressionWidth)
            contours = self.probabilityThresholder.process(maskFrame)
            if contours:
                contours.sort(key=lambda x: cv2i.contourArea(x),reverse=True)
                ellipse = cv2i.fitEllipse(contours[0])
                moments = cv2i.moments(contours[0])
                centroid = (moments['m10']/moments['m00'],moments['m01']/moments['m00'])
                if mask[-5] == "1":
                    centroid = (self.compressionWidth - centroid[0])/float(self.compressionWidth)
                    pointsRight.append([centroid, np.pi - np.radians(ellipse[2])])
                else:    
                    pointsLeft.append([centroid[0]/float(self.compressionWidth),np.radians(ellipse[2])])
        self.leftMeasurements.extend(pointsLeft)
        self.rightMeasurements.extend(pointsRight)
       

    def __setIdentificationFunctions__(self):
        from scipy.stats import gaussian_kde
        pointsLeft = np.array(self.leftMeasurements)
        pointsRight = np.array(self.rightMeasurements)
        densityLeft = gaussian_kde(pointsLeft.T, 2)
        densityRight = gaussian_kde(pointsRight.T, 2)
        self.leftFunction = densityLeft
        self.rightFunction = densityRight
