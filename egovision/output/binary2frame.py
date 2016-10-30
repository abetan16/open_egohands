__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['BinaryDecisionManager']

import sys, os
sys.path.append(os.path.abspath('../../'))
import numpy as np
from egovision.interfaces import cv2i


colorMapping = {0: (0,0,255), 1: (0,255,0)}
colorDash = (200,200,200,200) # nice to add a dash to the image

class BinaryDecisionManager:
    """

    The binaryDecisionManager is an object created to link binary decisions
    with the particular frame under analysis. This object is just a
    visualization link created for the hand-detection problem but implemented
    in general for binary classifiers. In the following example we compare a
    simple frame by frame and a dynamic hand-detector. For the example we use
    the VideoVisualizer and VideoWriter in a sequential fashion.

    .. |hd1| image:: ../_images/hand_detection/hd1.png
        :width: 200pt
    .. |hd2| image:: ../_images/hand_detection/hd2.png
        :width: 200pt
    .. |hd3| image:: ../_images/hand_detection/hd3.png
        :width: 200pt
    .. |hd4| image:: ../_images/hand_detection/hd4.png
        :width: 200pt
    .. |hd5| image:: ../_images/hand_detection/hd5.png
        :width: 200pt
    .. |hd6| image:: ../_images/hand_detection/hd6.png
        :width: 200pt

    +-+-------+-------+-------+-------+-------+-------+-+
    | | |hd1| + |hd2| + |hd3| + |hd4| + |hd5| + |hd6| | |
    +-+-------+-------+-------+-------+-------+-------+-+

    * Example 1 - Visualizing and comparing two hand detectors::
        
        from egovision import Video 
        from egovision.output import VideoVisualizer
        from egovision.handDetection import HandDetectionDataManager
        from egovision.handDetection import HandDetector
        from egovision.handDetection import DynamicHandDetector
        from egovision.output import BinaryDecisionManager
        from egovision.values.paths import VIDEO_EXAMPLE_PATH
        from egovision.values.paths import DATASET_HANDDETECTOR_GT_PATH as HD_PATH
        from egovision.interfaces import cv2i

        # Intializing relevant objects
        video = Video(VIDEO_EXAMPLE_PATH.format("BENCHTEST.MP4"))
        hd = HandDetector.load(HD_PATH.format("UNIGEmin","RGB","SVM",""))
        hdd = DynamicHandDetector.load(HD_PATH.format("UNIGEmin","RGB","SVM","_dynamic"))
        visualizer = VideoVisualizer()
        bdm = BinaryDecisionManager(["Frame by Frame","Dynamic"])

        # Visualizing the classification
        for nd, frame in enumerate(video):
            hands = hd.classifyFrame(frame,dtype="integer")
            handsd = hdd.classifyFrame(frame,dtype="integer")
            frame = bdm.decisions2frame(frame, [hands,handsd])
            visualizer.showFrameQuick("windowName",frame)
            if cv2.waitKey(1) > 1:
                break
            else:
                pass  

    * Example 2 - Export to a new video the comparsion of two hand detectors::

        from egovision import Video
        from egovision.handDetection import HandDetector
        from egovision.handDetection import DynamicHandDetector
        from egovision.output import VideoWriter
        from egovision.output import BinaryDecisionManager
        from egovision.values.paths import VIDEO_EXAMPLE_PATH
        from egovision.values.paths import DATASET_HANDDETECTOR_GT_PATH as HD_PATH
        from egovision.interfaces import cv2i

        # Intializing relevant objects
        video = Video(VIDEO_EXAMPLE_PATH.format("BENCHTEST.MP4"))
        hd = HandDetector.load(HD_PATH.format("UNIGEmin","RGB","SVM",""))
        hdd = DynamicHandDetector.load(HD_PATH.format("UNIGEmin","RGB","SVM","_dynamic"))
        writer = VideoWriter("BENCHTEST.avi")
        writer.setParametersFromVideo(video)
        bdm = BinaryDecisionManager(["Frame by Frame","Dynamic"])

        # Visualizing the classification
        for nd, frame in enumerate(video):
            hands = hd.classifyFrame(frame,dtype="integer")
            handsd = hdd.classifyFrame(frame,dtype="integer")
            frame = bdm.decisions2frame(frame, [hands,handsd])
            writer.writeFrame(frame)


    """
    
    def __init__(self,labelList,colorMapping = colorMapping):
        self.labelList = labelList
        self.colorMapping = colorMapping
        self.markLocations = []
        self.markHeight = 40
        self.markWidth = 40
        self.initialHeight = 60
        self.panel = []
        self.__defineLocations__()
    
    def __defineLocations__(self):
        self.markLocations = [(30,self.initialHeight + nx*(self.markHeight+10)) for nx, x in enumerate(self.labelList)]

    def __initializePanel__(self, frame, decisionList):
        self.panel = np.zeros(frame.matrix.shape, np.uint8)
        finalHeight = self.initialHeight + len(decisionList)*(self.markHeight + 10)
        cv2i.rectangle(self.panel, (0,self.initialHeight - 10),
                                  (500,finalHeight),
                                  (70,50,50),-1)

    def decisions2frame(self,frame, decisionList):
        """
            Frame is the frame object
            decision list is just the decision of one fram
        """
        if self.panel == []:
            self.__initializePanel__(frame,decisionList)
    
        frame.matrix = cv2i.addWeighted(frame.matrix, 1, self.panel, 1, 0)
        for nd, d in enumerate(decisionList):
            location = self.markLocations[nd]
            c1 = self.colorMapping[d[0]]
            p1 = (location[0], location[1])
            p2 = (location[0] + self.markWidth, location[1] +self.markHeight)
            cv2i.rectangle(frame.matrix, p1, p2, c1,-1)
            label = self.labelList[nd]
            location = self.markLocations[nd]
            p3 = (location[0] + self.markWidth + 10, location[1] + 3*self.markHeight/4)
            cv2i.putText(frame.matrix, label, p3, cv2i.FONT_HERSHEY_COMPLEX, 1, (60,60,60),2)
        return frame
