__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['VideoVisualizer']

import sys, os
sys.path.append(os.path.abspath('../'))
from egovision.interfaces import cv2i

class VideoVisualizer:
    """

    This class show the video in the computer screen. It is important to note
    that the video is not stream at the processing speed and not at the correct
    frames per second.
    
    :param String name: Name of window to display the video.


    * Example 1::
    
        from egovision import Video
        from egovision.values.paths import VIDEO_EXAMPLE_PATH
        from egovision.visualization import VideoVisualizer
        video = Video(VIDEO_EXAMPLE_PATH.format("BENCHTEST.MP4"))
        VideoVisualizer.showVideo("BENCHTEST", video)

    """

    @classmethod
    def showFrame(self, window, frame):
        """
        Display a fream.
        
        :param Frame frame: frame to be displayed.

        * Example 1::
        
            from egovision import Video
            from egovision.values.paths import VIDEO_EXAMPLE_PATH
            from egovision.output import VideoVisualizer
            video = Video(VIDEO_EXAMPLE_PATH.format("BENCHTEST.MP4"))
            VideoVisualizer.showFrame("BENCHTEST", video.next())
        """
        cv2i.imshow(window,frame.matrix)
        k = cv2i.waitKey(0)
        if k != -1:
            cv2i.destroyWindow(window)
            [cv2i.waitKey(1) for x in range(10)] # The worst solution of the world

    

    @classmethod
    def showFrameQuick(self, window, frame):
        """

        Display a frame withot waiting function. This is used to see results of
        the algorithms on the fly or to visualize videos frame by frame. If not
        cv2.waitKey is called then the window is not displayed.
        
        :param Frame frame: frame to be displayed.

        * Example 1::
        
            from egovision import Video
            from egovision.values.paths import VIDEO_EXAMPLE_PATH
            from egovision.output import VideoVisualizer
            video = Video(VIDEO_EXAMPLE_PATH.format("BENCHTEST.MP4"))
            VideoVisualizer.showFrameQuick("BENCHTEST", video.next())
        """
        cv2i.imshow(window, frame.matrix)
        for i in range(1):
            cv2i.waitKey(1)

    @classmethod
    def showVideo(self, window, video):
        """
        Display a video.
        
        :param Video video: Show video.

        * Example 1::
        
            from egovision import Video
            from egovision.values.paths import VIDEO_EXAMPLE_PATH
            from egovision.output import VideoVisualizer
            video = Video(VIDEO_EXAMPLE_PATH.format("BENCHTEST.MP4"))
            VideoVisualizer.showVideo("BENCHTEST", video)
        """
        for frame in video:
            self.showFrameQuick(window, frame)
            if cv2i.waitKey(1) > 0:
                break
            else:
                pass        
