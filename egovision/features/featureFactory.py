__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['FeatureAlgorithmFactory']

class FeatureAlgorithmFactory:
    
    def __init__(self):
        self.LOG_TAG = "[FeatureModel] "
    
    @classmethod
    def getAlgorithm(cls, feature):
        if feature == "GIST":
            from featureAlgorithms import GIST
            return GIST()
        elif feature == "HSV":
            from featureAlgorithms import HSV
            return HSV()
        elif feature == "YUV":
            from featureAlgorithms import YUV
            return YUV()
        elif feature == "LAB":
            from featureAlgorithms import LAB
            return LAB()
        elif feature == "RGB":
            from featureAlgorithms import RGB
            return RGB()
        elif feature == "RHL":
            from featureAlgorithms import RHL
            return RHL()
        elif feature == "HOG":
            from featureAlgorithms import HOG
            return HOG()
        elif feature in ["RGB-HIST", "HSV-HIST", "LAB-HIST", "YCrCb-HIST"]:
            from featureAlgorithms import ColorHistogram
            return ColorHistogram(feature)
