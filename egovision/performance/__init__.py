"""
EgoVision: Library to process Egocentric Videos.
"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['confusionMatrix','readGroundTruth','getFScore']



from confusionMatrix import getConfusionMatrix
from binaryPerformance import getFScore
from utils import readGroundTruth
