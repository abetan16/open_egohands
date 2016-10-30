""" Basic Objects of egoVision
"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['Video','VideoImg']


import numpy
from interfaces import cv2i 
import os
from egovision.frame import Frame

class Video:
    """
    
    Video object to encapsulate the opencv library. it stores the relevant
    information about the video as well as the opencv video capture. The Video
    object could be used also as an iterator of Frame objects.
    
    :param str filename: Video filename to be imported. It must include the \
    extension.

    :ivar str filename: filename.

    :ivar str LOG_TAG: Tag to be used for debuging purposes.

    :ivar opencv.stream stream: opencv stream object to read the video.

    """
    def __init__(self,filename):
        self.LOG_TAG = "[Video Object] "
        self.filename = filename
        if os.path.isfile(filename):
            self.stream = cv2i.VideoCapture(filename) #OPENCV stream
        else:
            msg = "This video file does not exist: " + filename
            from exceptions import InvalidVideo
            raise InvalidVideo(msg)

    def grab(self):
        return self.stream.grab()

    def retrieve(self):
        success, frame = self.stream.retrieve()
        return success, Frame(frame)

    def read(self):
        """
        
        Next function implemented as a requisite to use the object as an iterator.


        :returns: [Boolean, Frame]

        """
        success, matrix = self.stream.read()
        if success:
            frame = Frame(matrix)
        else:
            frame = None
        return success, frame
    
    def readFrame(self,frameNumber):
        """
        
        Return the frame in the particular frameNumber.
        
        """
        self.stream.set(cv2i.cv.CV_CAP_PROP_POS_FRAMES,int(frameNumber))
        return self.read()

    def release(self):
        """
        
        Release the videocapture.
        
        """
        self.stream.release()

    def getProperty(self, propertyName):
        """
        
        Return a property of the video. 
            

        :ivar String propertyName: Property name. The properties current available are:

            * number of frames
            
        """
        egovision2opencv = {
            "number of frames": lambda: int(self.stream.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)),
            "frames per second": lambda: int(self.stream.get(cv2.cv.CV_CAP_PROP_FPS))

        }
        if egovision2opencv.has_key(propertyName):
            return egovision2opencv[propertyName]()
        else:
            availableProperties = "\n".join(egovision2opencv.keys())
            msg = propertyFile + " is not supported yet. please try one of these:\n" + \
                  availableProperties
            from exceptions import InvalidProperty
            raise InvalidProperty(msg)


    
    def sampling2images(self,folder,frameDifference):
        """
        
        Export a unform sampling of the video to jpg images. 
            

        :ivar String folder: Folder to export the images.

        :ivar int frameDifference: Sampling steps, for example if frameDifference is \
        100, then the exported frames are {0,100,200,...}
 

        * Example 1: Sampling a video each 100 frames. To run this example is \
        required to create the output folder first::

            import egovision
            from egovision import Video

            DATASET = "UNIGE"
            VIDEO = "UNIGE_OFFICETEST.MP4"
            DATASET_PATH = "egovision/dataExamples/{0}/"
            OUTPUT_FOLDER = "results/sampling/"
            VIDEO_PATH = DATASET_PATH + "Videos/{1}"


            video = Video(VIDEO_PATH.format(DATASET,VIDEO))
            video.sampling2images(OUTPUT_FOLDER,100)


        """
        if folder[-1] != "/":
            folder += "/"
        videoName = self.filename.split("/")[-1]
        videoName = videoName.split(".")[0]
        FILENAME = folder + "{0}.jpg"
        numberOfFrames = self.getProperty("number of frames")
        for nf in range(0,numberOfFrames,frameDifference):
            self.stream.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,int(nf))
            frame = self.read()[1]
            frameString = str(nf).zfill(8)
            frame.exportAsImage(FILENAME.format(frameString))
            self.stream.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,int(0))
        

    def __iter__(self):
        return self

    def next(self):  
        """
        
        Next function implemented as a requisite to use the object as an iterator.
        
        :return: [Frame] Next frame in the video stream.

        """
        success, frame = self.read()
        if not success:
            raise StopIteration
        else:
            return frame




class VideoImg(Video):

    def __init__(self,imgFolder):
        self.LOG_TAG = "[Video Object] "
        self.imgFolder = imgFolder
        if os.path.isdir(imgFolder):
            self.currentFrame = 0
            self.FILE_NAME = imgFolder + "/{0}.jpg"
            self.lastFrame = len(os.listdir(imgFolder))
        else:
            msg = "This image does not exist: " + imgFolder
            from exceptions import InvalidVideo
            raise InvalidVideo(msg)


    def read(self):
        """
        
        Next function implemented as a requisite to use the object as an iterator.


        :returns: [Boolean, Frame]

        """
        success, frame = self.readFrame(self.currentFrame)
        self.currentFrame += 1
        return success, frame

    def readFrame(self,frameNumber):
        """
        
        Return the frame in the particular frameNumber.
        
        """
        if frameNumber < self.lastFrame:
            frameString = str(frameNumber).zfill(8)
            fileName = self.FILE_NAME.format(frameString)
            frame = Frame.fromFile(fileName)
            self.currentFrame = frameNumber + 1
            if frame:
                return True, frame
            else:
                return False, frame
        else:
            return False, None
