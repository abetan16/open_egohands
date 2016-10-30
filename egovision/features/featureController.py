"""
Feature extraction is one of the main steps in computer vision. This module encapsulated the feature extraction methods looking to reduce the impact of future implementations and to keep control of the inputs and outputs.  The feature module follows a Model-View-Controller design, on which the :ref:`FeatureController` receives the Video/Frame and the feature type (e.g. HoG, GIST, RGB, etc.), then ask the :ref:`FeatureModel` to extract the specific features, and finally returns a :ref:`FeatureVideo` (The View). As an extra functionality to speed up the experiment design process, the FeatureVideo can be saved and loaded to avoid repetitive feature extraction.

.. image:: ../_images/diagrams/features.png

Note from the figure that FeatureModel uses ObjectFactory to obtain the appropriate feature extraction algorithm. This factory is designed to reduce the impact of including new features or algorithmic changes in the feature extraction process. This factory design encapsulates the feature methods to the user which is never required to interact them directly.
"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['FeatureController']


import numpy as np
from featureModel import FeatureModel
from featureVideo import FeatureVideo
import cPickle
from egovision import Video, Frame
import sys,os

class FeatureController: 

    """ Reusable object to estimate features of a frame or a Video. The
    FeatureController is the interface to the feature module. To initialize the
    FeatureController it is necessary to specify the compression width and the
    type of feature. The compression rate is applied to resize each frame
    before the feature estimation. The list of features is growing quickly! by
    now there are Shape, Color and Global features.  For more information about
    performance of different color features, please refer to
    :cite:`Morerio2013`.
    
    :param float compressionWidth: Compression parameter that is applied to the \
    frame before the feature extraction.

    :param String feature: Feature to be extracted.

        .. list-table::
           :widths: 20 20 20
           :header-rows: 1

           * - Name
             - Command
             - type
           * - `RGB <http://en.wikipedia.org/wiki/RGB_color_model>`_
             - "RGB"
             - Color
           * - `YUV <http://en.wikipedia.org/wiki/YUV>`_
             - "YUV"
             - Color
           * - `LAB <https://en.wikipedia.org/wiki/Lab_color_space>`_
             - "LAB"
             - Color
           * - `HSV <https://en.wikipedia.org/wiki/Lab_color_space>`_
             - "HSV"
             - Color
           * - RHL (RGB + HSV + LAB)
             - "RHL"
             - Color
           * - GIST :cite:`Murphy2006`
             - "GIST"
             - Global
           * - HOG :cite:`Dalal2005`
             - "HOG"
             - Shapes
           * - Global Histograms Features
             - "<feature>-HIST"
             - Color Histograms

    :ivar str LOG_TAG: Tag to be used for debuging purposes

    :ivar FeatureModel featureModel: Attribute used to extract the feature of a \
    video or Frame

    """

    def __init__(self, compressionWidth, feature):
        

        self.LOG_TAG = "[FeatureController] "
        self.compressionWidth = compressionWidth
        self.featureModel = FeatureModel(feature)
        self.width = None
        self.height = None
    
    def __getFrameFeature__(self, frame):
        frame = frame.resizeByWidth(self.compressionWidth)
        frameMatrix = frame.matrix
        self.width = frameMatrix.shape[1]
        self.height = frameMatrix.shape[0]
        feature = self.featureModel.getFeature(frameMatrix)
        return feature

    def getFeatures(self, sequence):

        """ Estimates the desired feature of the video/frame object. Returns a
        FeatureVideo with the estimated features.

        :param Frame/Video sequence: Frame or Video object.
      
        :returns: [FeatureVideo] Return a FeatureVideo that can be easily saved \
        to speed up your computational experiments.
        
        """    
        
        featureView = FeatureVideo(self.featureModel.featureType)
        if isinstance(sequence, Frame):
            desc = self.__getFrameFeature__(sequence)
            featureView.append(desc)
            return True, featureView
        elif isinstance(sequence, Video):
            for nf, frame in enumerate(sequence):
                desc = self.__getFrameFeature__(frame)
                featureView.append(desc)
            featureView.features = np.vstack(featureView.features)
            return True, featureView
        else:
            from exceptions import UndefinedSequence
            raise UndefinedSequence("Impossible to get features from this \
type of sequence")
            return False, []

