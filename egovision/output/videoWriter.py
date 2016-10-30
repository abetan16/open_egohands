__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['VideoWriter']

import sys, os
sys.path.append(os.path.abspath('../'))
from egovision.interfaces import cv2i



class VideoWriter:
    """

    This is an interface to video writer class of opencv. Using this class is
    possible to export a full video in a sequential fashion or using procesed
    frames.

    This method only export avi videos.

    :param String filename: Name of the file to be written.

    :ivar tuple outputSize: (height, witht) Tuple of integers representing the\
    size of the frames. Default value (180, 320).

    :ivar int fps: (height, witht) Tuple of integers representing the\
    size of the frames. Default value 50.

    * Example 1::

        from egovision import Video
        from egovision.values.paths import VIDEO_EXAMPLE_PATH
        from egovision.output import VideoWriter
        video = Video(VIDEO_EXAMPLE_PATH.format("BENCHTEST.MP4"))
        writer = VideoWriter("BENCHTEST.avi")
        writer.setParametersFromVideo(video)
        writer.exportVideo(video)

    """

    def __init__(self,filename,outputSize=(180,320), fps=50) :
        self.filename = filename
        self.streamOut = cv2i.VideoWriter()
        self.outputSize = (outputSize[1],outputSize[0])
        self.framesPerSecond = fps
        self.codec = cv2i.cv.CV_FOURCC('D','I','V','X')
    
    def setParametersFromVideo(self,video):
        """
        Set the parameters using an already imported video.
        
        :param Video video: Video with the desired parameters.

        * Example 1::
        
            from egovision import Video
            from egovision.values.paths import VIDEO_EXAMPLE_PATH
            from egovision.output import VideoWriter
            video = Video(VIDEO_EXAMPLE_PATH.format("BENCHTEST.MP4"))
            writer = VideoWriter("BENCHTEST.avi")
            writer.setParametersFromVideo(video)
        """

        height = int(video.stream.get(cv2i.cv.CV_CAP_PROP_FRAME_HEIGHT))
        width = int(video.stream.get(cv2i.cv.CV_CAP_PROP_FRAME_WIDTH))
        self.codec = cv2i.cv.CV_FOURCC('D','I','V','X') # 130
        try:
            self.framesPerSecond = int(video.stream.get(cv2i.cv.CV_CAP_PROP_FPS))
        except:
            self.framesPerSecond = 60
        self.outputSize = (width, height)

    def writeFrame(self, frame):
        """

        Write a frame. It is important to release the VideoWriter once the
        writing process is finished.
        
        :param Frame frame: Frame to be written.

        * Example 1::
        
            from egovision import Video
            from egovision.values.paths import VIDEO_EXAMPLE_PATH
            from egovision.output import VideoWriter
            video = Video(VIDEO_EXAMPLE_PATH.format("BENCHTEST.MP4"))
            writer = VideoWriter("BENCHTEST.avi")
            writer.setParametersFromVideo(video)
            writer.writeFrame(video.next())
            writer.release()
        """
        if not self.streamOut.isOpened():
            self.streamOut.open(self.filename, self.codec, self.framesPerSecond, self.outputSize, True)
        self.streamOut.write(frame.matrix)
    
    def exportVideo(self,video):
        """
        
        Export a video object to avi format.
        
        :param Video video: Video to export.

        * Example 1::
        
            from egovision import Video
            from egovision.values.paths import VIDEO_EXAMPLE_PATH
            from egovision.output import VideoWriter
            video = Video(VIDEO_EXAMPLE_PATH.format("BENCHTEST.MP4"))
            writer = VideoWriter("BENCHTEST.avi")
            writer.setParametersFromVideo(video)
            writer.exportVideo(video)


        """
        self.streamOut.open(self.filename, self.codec, self.framesPerSecond, self.outputSize, True)
        for nf, frame in enumerate(video):
            sys.stdout.write("\r{0}>".format(nf))
            sys.stdout.flush()
            self.writeFrame(frame)
        self.release()

    def release(self):
        self.streamOut.release()
