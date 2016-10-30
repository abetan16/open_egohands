""" Basic Objects of egoVision
"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['Frame']


import numpy
from interfaces import cv2i
import os

class Frame:
    """

    Frame instance for video processing. Frame encapsulates the opencv and\
    numpy frames and guarantee that the Frame.matrix is allways a numpy ndarray\
    of integers 8 [uint8]\
    
    :param Matrix matrix: Matrix representation to be standarized and stored as a\
    Frame object.

    :param str pickleName: Optional filename containing a previously saved \
    frame. The default value is None but if pickleName is used then it will\
    return the object of the pickle.\

    :ivar numpy.array matrix: Standarized representation of the frame

    :ivar str LOG_TAG: Tag to be used for debuging purposes

    """
    def __init__(self, matrix):
        self.LOG_TAG = "[Frame object] "
        if isinstance(matrix, (numpy.matrix,numpy.ndarray)):
            self.matrix = matrix # Matix representation of the frame
        elif isinstance(matrix, list):
            
            self.matrix = numpy.array(matrix,dtype="uint8")
        else:
            print self.LOG_TAG + str(matrix)  + "This kind of object is not allowed as a frame"


    @classmethod
    def loadMask(cls, filename, compressionWidth=None):
        """
        
        Loading a amas and returning an binary mask frame

        :ivar String filename: filename

        :ivar String frameType: [RBG|GRAYSCALE]
        
        """
        import cv
        obj = cls([])
        obj.matrix = cv2i.imread(filename,cv.CV_LOAD_IMAGE_GRAYSCALE)
        obj = obj.resizeByWidth(compressionWidth)
        obj.matrix = cv2i.threshold(obj.matrix, 50, 1, cv2i.THRESH_BINARY)[1]
        obj.matrix = numpy.reshape(obj.matrix,(obj.matrix.shape[0], obj.matrix.shape[1], 1))
        return obj

    @classmethod
    def fromFile(cls, filename, frameType="BGR", compressionWidth=None):
        """
        
        Loading the object object as a python pickle file

        :ivar String filename: filename

        :ivar String frameType: [RBG|GRAYSCALE]
        
        """
        obj = cls([])
        if frameType == "BGR":
            obj.matrix = cv2i.imread(filename)
        elif frameType == "GRAYSCALE":
            obj.matrix = cv2i.imread(filename,cv.CV_LOAD_IMAGE_GRAYSCALE)
        obj = obj.resizeByWidth(compressionWidth)
        return obj


    def resize(self, height, width):
        """ Resizing the frame using a target height and widht.

        :ivar float height: target height

        :ivar float width: target width

        :returns: [Frame] Resized frame, The original frame is not modified to \
        avoid coding conflicts
        
        """
        newFrame = Frame(cv2i.resize(self.matrix, (int(width), int(height))))
        if len(self.matrix.shape) == 3 and self.matrix.shape[-1] == 1:
            newFrame.matrix = numpy.reshape(newFrame.matrix, 
                                                (int(height), int(width), 1))
        return newFrame

    def normalize(self):
        minValue = numpy.min(self.matrix)
        maxValue = numpy.max(self.matrix)
        scaled = (self.matrix - minValue)/(maxValue-minValue)
        return Frame(scaled)

    def fromBGR2ColorSpace(self,colorSpace):
        """
        
        Change the color space of the frame.

        :ivar String frameType: [RGB|HSV|LAB]

        :returns Frame: Copy of the frame in the new color space
        
        """
        if colorSpace == "LAB":
            return Frame(cv2i.cvtColor(self.matrix, cv2i.COLOR_BGR2LAB))
        if colorSpace == "RGB":
            return Frame(cv2i.cvtColor(self.matrix, cv2i.COLOR_BGR2RGB))
        elif colorSpace == "HSV":
            return Frame(cv2i.cvtColor(self.matrix, cv2i.COLOR_RGB2HSV))


    def resizeByWidth(self,width):
        """
        
        Resizing the frame using a target widht.

        :ivar float width: target width

        :returns: [Frame] Resized frame, The original frame is not modified to \
        avoid coding conflicts
        
        """

        if width:
            compressionRate = width/float(self.matrix.shape[1])
            newFrame = self.resizeByCompression(compressionRate)
            return newFrame
        else:
            return self

    def resizeByCompression(self,compression):
        """
        
        Resizing the frame using a compression percentage.

        :ivar float compression: compression rate in terms of the actual size
        
        :returns: [Frame] Resized frame, The original frame is not modified to \
        avoid coding conflicts

        """
        width = self.matrix.shape[1]*compression
        height = self.matrix.shape[0]*compression
        newFrame = self.resize(height,width)
        return newFrame


    def exportAsImage(self,filename):
        """
        
        Export the frame as an image
 
        :param String filename: Filename with extension.

        """
        cv2i.imwrite(filename,self.matrix)

    def __eq__(self,other):
        """
        
        Logic operator to define if two frames are equal.
        
        """
        if (self.matrix.shape == other.matrix.shape):
            return True
        else:
            return False

