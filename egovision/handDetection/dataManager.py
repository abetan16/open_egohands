__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['HandDetectorDataManager']

import sys,os
sys.path.append(os.path.abspath('../'))
import numpy
from egovision.extras import fullVideoName


class HandDetectionDataManager():

    """ HandDetectionDataManager is an object designed to store the features
    and attributes to be used in the trainning stage. It could be thought as
    the training dataset. This oject allows you to read the hand-detection
    datasets and keep control of the training parameters to improve the
    replicability of your results. 
    
    Initially the handDetectionDataManager initialize its attributes as empty
    lists; however, once  the  method  readDataset  is  called the attributes
    are stored for future use.  The main  objective of this object is to
    simplify the training process reducing the feature extraction, for example
    multiple type of HandDetectors could be trained using the same
    HandDetectionDataManager. If used properly it could reduce your
    computational time when developing computational experiments based on the
    same dataset and features.

    :ivar str LOG_TAG: Tag to be used for debuging purposes

    :ivar list headers: VideoName and frame number in the same order as the\
    attributes.

    :ivar list attributes: extracted features.

    :ivar list categories: Categorical variables per frame, where 1 is a frame\
    with hands and 0 a frame without hands.

    :ivar string datasetFolder: Foldername containing the training frames. For\
    a more detailed structure of the dataset (see :func:`egovision.handDetection.HandDetectionDataManager.readDataset`)

    * Example 1: Reading it from the dataset folder and saving it as a pickle::

            from egovision.handDetection import HandDetectionDataManager
            from egovision.values.paths import DATASET_PATH
            from egovision.extras import ObjectPickler
            feature = "HOG"
            dataset = "UNIGEmin"
            datasetFolder = DATASET_PATH.format(dataset)
            dm = HandDetectionDataManager()
            dm.readDataset(datasetFolder, 200, feature)
            success = ObjectPickler.save(dm, "test_dm.pk")    

    * Example 2: Loading from a pickle file::

            from egovision.handDetection import HandDetectionDataManager
            from egovision.extras import ObjectPickler
            dm = ObjectPickler.load(HandDetectionDataManager, "test_dm.pk")
    """

    def __init__(self):
        self.LOG_TAG = "[HandDetectionDataManager] "
        self.headers = []
        self.attributes = []
        self.categories = []
        self.datasetFolder = None
        self.feature = None
        self.compression = None
 
    def readDataset(self,datasetFolder, compressionWidth, feature):
        
        """

        This method reads the folder structure of the dataset and initialize
        the attributes of the data manager. In general the folder structure is
        divided in three parts: i) Videos: contains the raw video sequences,
        ii) Positives: Containing the masks of the positive samples, iii)
        Negatives: Containing the masks of the negative samples. For
        illustrative purposes lets name our dataset as "EV", and lets define
        its root folder as "EV/". The folder structure is briefly summarized in
        the next table:

        .. list-table::
           :widths: 10 20 60
           :header-rows: 1
           
           * - Path
             - Content
             - Description 
           * - <dataset>/Videos
             - Full video files
             - Original video sequences. Each video could contains positives as
               well as negative frames. Each video should be named as
               <dataset>_<videoid>.<extension>. For example the full path of a
               video in the EV dataset could be "EV/Videos/EV_Video1.MP4".  
           * - <dataset>/Positives
             - Folders
             - <dataset>/Positives contains a folder per video that is going to
               be used to extract positive frames (with hands). For example,
               lets assume that the frame 10, 20 and 30 of
               "EV/Videos/EV_Video1.MP4" are going to be used as positive
               samples in the training stage, then the positives folder should
               contain these files::
               
                    "EV/Positives/EV_Video1/mask10.jpg",
                    "EV/Positives/EV_Video1/mask20.jpg",
                    "EV/Positives/EV_Video1/mask30.jpg"

               respectively. In practice the mask files could be empty files
               because they are used only to guide the scanning of the video.
               However, as a way to validate the used frames we suggest to use
               compressed snapshots of the real frames.
           * - <dataset>/Negatives
             - Folders
             - <dataset>/Negatives contains a folder per video that is going to
               be used to extract negative frames (without hands). For example,
               lets assume that the frame 30, 100 and 120 of
               "EV/Videos/EV_Video2.MP4" are going to be used as negative 
               samples in the training stage. To do this the negatives folder should
               contain these files::

                    "EV/Positives/EV_Video1/mask30.jpg",
                    "EV/Positives/EV_Video2/mask100.jpg",
                    "EV/Positives/EV_Video2/mask120.jpg" 
            
               respectively. In practice the mask files could be empty files
               because they are only used to guide the scanning of the video.
               However, as a way to validate the used frames we suggest to use
               compressed snapshots of the real frames.

        Finally, following the previous example the folder structure is::

            EV/Videos/EV_Video1.MP4
            EV/Videos/EV_Video2.MP4
            EV/Positives/EV_Video1/mask10.jpg
            EV/Positives/EV_Video1/mask20.jpg
            EV/Positives/EV_Video1/mask30.jpg
            EV/Negatives/EV_Video2/mask30.jpg
            EV/Negatives/EV_Video2/mask100.jpg
            EV/Negatives/EV_Video2/mask120.jpg


        Example 1: How to read the dataset folder from egovision::
        
            from egovision.handDetection import HandDetectionDataManager
            from egovision.values.paths import DATASET_PATH
            feature = "HOG"
            dataset = "UNIGEmin"
            datasetFolder = DATASET_PATH.format(dataset)
            dm = HandDetectionDataManager()
            dm.readDataset(datasetFolder, 200, feature)

        """
        from datetime import datetime
        from egovision import Video
        from egovision.features import FeatureController
        self.datasetFolder = datasetFolder
        self.compressionWidth = compressionWidth
        self.feature = feature
        categories = ["Negatives","Positives"]
        featureController = FeatureController(compressionWidth, feature)
        for nc, cat in enumerate(categories):
            categoryFolder = "".join([datasetFolder,"/",cat,"/"])
            videoNames = os.listdir(categoryFolder)
            for videoName in videoNames:
                masks = os.listdir("".join([categoryFolder,videoName]))
                masks.sort(key=lambda x: int(x[4:-4]))
                fVideoName = "".join([datasetFolder,"/Videos/",videoName])
                fVideoName = fullVideoName(fVideoName)
                video = Video(fVideoName)
                for mask in masks:
                    sys.stdout.flush()
                    fmNumber = int(mask[4:-4])
                    t0 = datetime.now()
                    success, frame = video.readFrame(fmNumber)
                    t1 = datetime.now()
                    success, desc = featureController.getFeatures(frame)
                    t2 = datetime.now()
                    # sysout = "\r{0}: - {1} - {2} - {3}".format(videoName, mask, t2-t1,t1-t0)
                    # sys.stdout.write(sysout)
                    self.headers.append("".join([fVideoName,"_",str(fmNumber)]))
                    self.attributes.append(desc.next())
                    self.categories.append(nc)
        self.attributes = numpy.vstack(self.attributes)

    def readFromVideoAndFrames(self, videoName, frames, category, compression = 0.16, feature="HOG"):
        from egovision import Video
        from egovision.features import FeatureController
        import time
        featureController = FeatureController(compression)
        video = Video(videoName)
        nf = 0
        while frames != []:
            success = video.grab()
            if int(frames[0]) == nf:
                success, frame = video.retrieve()
                t0 = time.time()
                desc = featureController.getFeature(frame,feature)
                t1 = time.time()
                self.headers.append("".join([videoName,"_",str(nf)]))
                self.attributes.append(desc)
                self.categories.append(category)
                frames.pop(0)
            else:
                pass
            nf += 1
        self.attributes = numpy.vstack(self.attributes)

    def mergeDataManager(self,dt):
        self.headers.extend(dt.headers)
        if self.attributes != []:
            self.attributes = numpy.vstack((self.attributes,dt.attributes))
        else:
            self.attributes = dt.attributes
        self.categories.extend(dt.categories)

    def subDataTraining(self,index):
        dt = DataTraining()
        dt.headers = [self.headers[x] for x in index]
        dt.attributes = [self.attributes[x] for x in index]
        dt.categories = [self.categories[x] for x in index]
        return dt
        

if __name__ == "__main__":
    import doctest
    doctest.testmod()

