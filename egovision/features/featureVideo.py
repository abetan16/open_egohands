__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['FeatureVideo']

import numpy
import cPickle


class FeatureVideo:
    """

    This is objects plays the role of the View component in the proposed MVC
    design. It can be considered an iterator of :class:`Feature` objects or as
    the feature sequence extracted from a video. This object is helpful when
    designing the experiments because it allows the data scientist to extract
    expensive features only once and save the FeatureVideo for future uses.

    :param float compressionWidth: Proportion of the original width to be \
    used in the resizing stage before estimate the features. If compression \
    rate is 1 then the original image is used [Default = 0.2].

    :param String featureType: Feature to be used.

    :ivar str LOG_TAG: Tag to be used for debuging purposes

    :ivar List<Feature> features: List with the extracted Features

    :ivar List<Feature> features: Pointer to the current position

    """
    def __init__(self, featureType):
        self.LOG_TAG = "[FeatureVideo] "
        self.features = [] # List of lists
        self.current = 0
        self.featureType = featureType
    
    def append(self, feature):
        """
        Append new feature at the end of the FeatureVideo

        :param feature: Features to be appended
    
        """
        self.features.append(feature)


    def __iter__(self):
        return self

    def next(self):
        """
        
        Next function implemented as a requisite to use the object as an iterator.
        
        :return: [Frame] Next frame in the video stream.

        """
        if self.current > len(self.features):
            raise StopIteration
        else:
            self.current += 1
            return self.features[self.current-1]
