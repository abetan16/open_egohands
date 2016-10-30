"""
EgoVision: Library to process Egocentric Videos.
"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['Video','VideoImg','Frame','handDetection','handSegmentation','features','output','performance', 'extras']

from video import Video, VideoImg
from frame import Frame
import handDetection
import features
import output
import performance
import handSegmentation
import extras
