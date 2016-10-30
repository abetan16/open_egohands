import unittest
import sys,os
sys.path.append(os.path.abspath('../'))
from egovision.frame import Frame
from egovision.video import Video
from egovision.handDetection import HandDetectionDataManager
from egovision.handDetection import HandDetector
from egovision.handDetection import DynamicHandDetector
from egovision.values.paths import VIDEO_EXAMPLE_PATH
from egovision.values.paths import DATASET_PATH
from egovision.values.paths import DATASET_DATAMANAGER_PATH
from egovision.values.paths import DATASET_DATAMANAGER_GT_PATH
from egovision.values.paths import DATASET_HANDDETECTOR_GT_PATH 
from egovision.values.paths import GROUNDTRUTH_VIDEOFEATURE_PATH
from egovision.features import FeatureVideo
from egovision.extras import ObjectPickler
import numpy
from test_parameters import createGroundTruths

if os.getcwd().endswith("handDetection"):
    DATASET_PATH = "../" + DATASET_PATH
    VIDEO_EXAMPLE_PATH = "../" + VIDEO_EXAMPLE_PATH
    DATASET_DATAMANAGER_PATH = "../" + DATASET_DATAMANAGER_PATH
    DATASET_DATAMANAGER_GT_PATH = "../" + DATASET_DATAMANAGER_GT_PATH
    DATASET_HANDDETECTOR_GT_PATH = "../" + DATASET_HANDDETECTOR_GT_PATH
    GROUNDTRUTH_VIDEOFEATURE_PATH = "../" + GROUNDTRUTH_VIDEOFEATURE_PATH

class DataManagerTestCase(unittest.TestCase):
    # THIS DATAMANAGER IS ONLY FOR TRAINING AND IS COMPOSED ONLY BY TRAINING FRAMES
    def __init__(self,methodName="runTest",feature=None,dataset=None):
        super(DataManagerTestCase, self).__init__(methodName)
        self.feature = feature
        self.dataset = dataset

    def setUp(self):
        pass

    def testDataManager(self):
        datasetFolder = DATASET_PATH.format(self.dataset)
        if os.path.exists(datasetFolder):
            dm = HandDetectionDataManager()
            dm.readDataset(datasetFolder,180,feature=self.feature)
            if createGroundTruths:
                print "[Hand Detection Data Manager] Ground Truth Created"
                gtfile = DATASET_DATAMANAGER_GT_PATH.format(self.dataset,self.feature)
                print gtfile
                if not os.path.exists(os.path.split(gtfile)[0]):
                    os.makedirs(os.path.split(gtfile)[0])
                success = ObjectPickler.save(dm, gtfile)    

            dm2 = ObjectPickler.load(HandDetectionDataManager, DATASET_DATAMANAGER_GT_PATH.format(self.dataset,self.feature))
            numpy.testing.assert_equal(dm.attributes,dm2.attributes)
            self.assertIsInstance(dm.attributes, numpy.ndarray,
                                  msg="New HandDetectionDataManager does not guarantee ndarray")
            self.assertIsInstance(dm2.attributes, numpy.ndarray,
                                  msg="GroundTruth is not ndarray")
        else:
            raise os.error("The dataset does not exists")

    def __str__(self):
        return "".join(["Data Manager from Dataset Folder: ",
                        self.dataset, " - ",
                        self.feature])


class HandDetectorTestCase(unittest.TestCase):
    # THIS TEST TRAINS THE CLASSIFIER AND COMPARES WITH THE GROUND TRUTH
    def __init__(self,methodName="runTest",dataset=None,
                                           feature=None,
                                           classifier=None,
                                           videotest=None,
                                           dynamic=""):
        super(HandDetectorTestCase, self).__init__(methodName)
        self.dataset = dataset
        self.feature = feature
        self.classifier = classifier
        if dynamic == "":
            self.handDetector = HandDetector(self.feature, 180,"SVM")
        else:
            self.handDetector = DynamicHandDetector(self.feature, 180,"SVM")
        self.videotest = videotest
        self.dynamic = dynamic
        
    def setUp(self):
        pass
    
    def testTrainingDetector(self):
        datasetFolder = DATASET_DATAMANAGER_GT_PATH.format(self.dataset,self.feature)
        dm = ObjectPickler.load(HandDetectionDataManager, datasetFolder)
        self.handDetector.trainClassifier(dm)
        handDetectorFile = DATASET_HANDDETECTOR_GT_PATH.format(self.dataset,
                                                               self.feature,
                                                               self.classifier,
                                                               self.dynamic)
        if createGroundTruths:
            print "[Hand Detector] Ground Truth Created"
            print handDetectorFile
            if not os.path.exists(os.path.split(handDetectorFile)[0]):
                os.makedirs(os.path.split(handDetectorFile)[0])
            success = ObjectPickler.save(self.handDetector, gtfile)    

        if self.dynamic == "":
            hd = ObjectPickler.load(HandDetector, handDetectorFile)
        else:
            hd = ObjectPickler.load(DynamicHandDetector, handDetectorFile)
            hd.setOptimalParameters(50.0)

        featureVideoPath = GROUNDTRUTH_VIDEOFEATURE_PATH.format(self.videotest, self.feature)
        featureVideo = ObjectPickler.load(FeatureVideo, featureVideoPath)
        featureVideo.features = featureVideo.features[:500]
        r1 = self.handDetector.classifyFeatureVideo(featureVideo,dtype="integer")
        r2 = hd.classifyFeatureVideo(featureVideo,dtype="integer")
        numpy.testing.assert_array_almost_equal(r1,r2,)
        if self.classifier == "SVM":
            if self.dynamic == "_dynamic": #restart the dynamic properties
                self.handDetector = DynamicHandDetector(self.feature, 180,"SVM")
                self.handDetector.trainClassifier(dm)
            r3 = self.handDetector.classifyFeatureVideo(featureVideo,dtype='float')
            r4 = self.handDetector.binarizeDetections(r3,th=0)
            numpy.testing.assert_array_almost_equal(r1,r4)
        self.assertIsInstance(r1, numpy.ndarray,
                              msg="Trained Hand detector is not returning arrays")
        self.assertIsInstance(r2, numpy.ndarray,
                              msg="Ground truth Hand detector is not returning arrays")

    def __str__(self):
        return "".join(["Feature V. Hand Detection Test ", self.dynamic, ": ",
                        self.dataset, " - ",
                        self.feature, " - ",
                        self.classifier])

class HandDetectorTestCaseFrame(unittest.TestCase):
    # THIS TEST USES ONE FRAME
    def __init__(self,methodName="runTest",dataset=None,
                                           feature=None,
                                           classifier=None,
                                           dynamic=""
                                           ):
        super(HandDetectorTestCaseFrame, self).__init__(methodName)
        self.feature = feature
        self.dataset = dataset
        self.classifier = classifier
        self.dynamic = dynamic
        
    def testClassificationFrame(self):
        handDetectorFile = DATASET_HANDDETECTOR_GT_PATH.format(self.dataset,
                                                               self.feature,
                                                               self.classifier,
                                                               self.dynamic)
        if self.dynamic == "":
            self.handDetector = ObjectPickler.load(HandDetector, handDetectorFile)
        else:
            self.handDetector = ObjectPickler.load(DynamicHandDetector, handDetectorFile)
            self.handDetector.setOptimalParameters(50.0)
        frame = ObjectPickler.load(Frame, VIDEO_EXAMPLE_PATH.format("frameMatrix.pk"))
        hands = self.handDetector.classifyFrame(frame,dtype="integer")
        self.assertIsInstance(hands, numpy.ndarray,
                              msg="Hand detector is not returning arrays")
        self.assertEqual(hands.size, 1,
                              msg="Hand detector is not returning arrays of size 1")
        self.assertIsInstance(hands[0], int,
                              msg="Hand detector is not returning arrays of size 1 with an integer")

        hands = self.handDetector.classifyFrame(frame,dtype="float")
        self.assertIsInstance(hands, numpy.ndarray,
                              msg="Hand detector is not returning arrays")
        self.assertEqual(hands.size, 1,
                              msg="Hand detector is not returning arrays of size 1")
        self.assertIsInstance(hands[0], float,
                              msg="Hand detector is not returning arrays of size 1 with a float")

    def __str__(self):
        return "".join(["Frame Hand Detection Test ",self.dynamic, ": ",
                        self.feature, " - ",
                        self.classifier])

class HandDetectorTestCaseVideo(unittest.TestCase):
    # THIS TEST USES THE VIDEO
    def __init__(self,methodName="runTest",dataset=None,
                                           feature=None,
                                           classifier=None,
                                           dynamic="",
                                           ):
        super(HandDetectorTestCaseVideo, self).__init__(methodName)
        self.feature = feature
        self.dataset = dataset
        self.classifier = classifier
        self.dynamic = dynamic


    def testClassificationVideo(self):
        handDetectorFile = DATASET_HANDDETECTOR_GT_PATH.format(self.dataset,
                                                               self.feature,
                                                               self.classifier,
                                                               self.dynamic)
        self.handDetector = ObjectPickler.load(HandDetector, handDetectorFile)
        video = Video(VIDEO_EXAMPLE_PATH.format("BENCHTEST.MP4"))
        hands = self.handDetector.classifyVideo(video,dtype="integer")
        self.assertIsInstance(hands, numpy.ndarray,
                              msg="Hand detector is not returning arrays")
        video.release()
        
    def __str__(self):
        return "".join(["Video Hand Detection Test: ",
                        self.feature, " - ",
                        self.classifier])            
        


def load_tests(loader, tests, pattern):
    from egovision.test_parameters import testingModules
    from test_parameters import features
    from test_parameters import classifiers
    from test_parameters import videoTest
    from test_parameters import extension
    from test_parameters import testingDataset
    from test_parameters import fast
    from test_parameters import dataManagerTraining
    from test_parameters import detectorTraining

    suite = unittest.TestSuite()
    

    if "handDetection" in testingModules:

        if not fast and dataManagerTraining:
            # Testing if is possible to read the dataset folder and create the DataManager
            for feature in features:
                suite.addTest(DataManagerTestCase('testDataManager',feature,testingDataset))


        # Testing if a new training is close to the ground truth
        if not fast and detectorTraining:
            for feature in features:
                for classifier in classifiers:
                    suite.addTest(HandDetectorTestCase('testTrainingDetector',testingDataset,feature,classifier,videoTest,"_dynamic"))
                    suite.addTest(HandDetectorTestCase('testTrainingDetector',testingDataset,feature,classifier,videoTest,""))


        # Testing if the hand detector is able to run in a video
        for feature in features:
            for classifier in classifiers:
                suite.addTest(HandDetectorTestCaseVideo('testClassificationVideo',testingDataset,feature,classifier,""))
                suite.addTest(HandDetectorTestCaseVideo('testClassificationVideo',testingDataset,feature,classifier,"_dynamic"))


        # Testing if the handDetector is able to run in a Frame object
        for feature in features:
            for classifier in classifiers:
                suite.addTest(HandDetectorTestCaseFrame('testClassificationFrame',testingDataset,feature,classifier,""))
                suite.addTest(HandDetectorTestCaseFrame('testClassificationFrame',testingDataset,feature,classifier,"_dynamic"))

        

    return suite

if __name__ == "__main__":
    unittest.main()
