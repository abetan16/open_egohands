__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['FeatureModel']

from feature import Feature
from featureFactory import FeatureAlgorithmFactory

class FeatureModel:
    """
    
    This object connects the FeatureController with the Feature Algorithm. 

    :param String feature: Feature to used.

    :ivar String LOG_TAG: Tag to be used for debuging purposes

    :ivar String featureType: Name of the used feature

    :ivar FeatureAlgorithm featureAlgorithm: Specific instance of the feature \
    extraction algorithm.


    """
    def __init__(self, featureType):
        self.LOG_TAG = "[FeatureModel] "
        self.featureAlgorithm = FeatureAlgorithmFactory.getAlgorithm(featureType)
        self.featureType = featureType
    
    def getFeature(self, frame):
        """ Estimates the desired feature of the video/frame object. Returns a
        Feature Object.

        :param Frame sequence: Frame object.
      
        :returns: [Feature] Extracted Feature.
        
        """    
        return Feature(self.featureAlgorithm.get(frame))
        
