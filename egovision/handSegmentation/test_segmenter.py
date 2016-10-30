import unittest
import numpy
import sys,os
sys.path.append(os.path.abspath('../'))
from egovision import Video
from egovision import Frame
from egovision.values.paths import DATASET_PATH
from egovision.values.paths import DATASET_MASKS_PATH
from egovision.values.paths import DATASET_FRAMES_PATH
from egovision.values.paths import DATASET_HSDATAMANAGER_GT_PATH
from egovision.values.paths import DATASET_MULTIHSDATAMANAGER_GT_PATH
from egovision.extras import ObjectPickler
from egovision.interfaces import cv2i
from egovision.modelRecommenders import NearestNeighbors

from test_parameters import dataset
from test_parameters import trainingVideo
from test_parameters import trainingMask
from test_parameters import createGroundTruths
from test_parameters import feature

from postProcessor import PostB2016

from dataManager import MaskBasedDataManager
from pixelByPixelHandSegmenter import PixelByPixelHandSegmenter
from pixelByPixelMultiHandSegmenter import PixelByPixelMultiHandSegmenter
from egovision.handSegmentation import SegmenterEvaluator

if os.getcwd().endswith("handSegmentation"):
    DATASET_PATH = "../" + DATASET_PATH
    DATASET_MASKS_PATH = "../" + DATASET_MASKS_PATH
    DATASET_FRAMES_PATH = "../" + DATASET_FRAMES_PATH
    DATASET_HSDATAMANAGER_GT_PATH = "../" + DATASET_HSDATAMANAGER_GT_PATH
    DATASET_MULTIHSDATAMANAGER_GT_PATH = "../" + DATASET_MULTIHSDATAMANAGER_GT_PATH



class MaskBasedDataManagerTestCase(unittest.TestCase):

    def __init__(self,methodName="runTest",feature=None):
        super(MaskBasedDataManagerTestCase, self).__init__(methodName)
        self.feature = feature

    def setUp(self):
        pass

    def runTest(self):
        mask = DATASET_MASKS_PATH.format(dataset, trainingVideo) + trainingMask
        gtfile = DATASET_HSDATAMANAGER_GT_PATH.format(dataset, trainingVideo, trainingMask, self.feature)
        dm = MaskBasedDataManager()
        dm.readDataset([mask],200,"LAB")
        if createGroundTruths:
            print "[Feature Creator] Ground Truth Created"
            print gtfile
            if not os.path.exists(os.path.split(gtfile)[0]):
                os.makedirs(os.path.split(gtfile)[0])
            ObjectPickler.save(dm, gtfile)
        dm2 = ObjectPickler.load(MaskBasedDataManager, gtfile)
        numpy.testing.assert_equal(dm.attributes, dm2.attributes)
        self.assertIsInstance(dm.attributes, numpy.ndarray,
                              msg="New HandDetectionDataManager does not guarantee ndarray")
        self.assertIsInstance(dm2.attributes, numpy.ndarray,
                              msg="GroundTruth is not ndarray")

    def __str__(self):
        return "".join(["Testing Mask Based Data Manager: ",
                        self.feature])

class SingleModelHandSegmenterTestCase(unittest.TestCase):

    def __init__(self,methodName="runTest",feature=None):
        super(SingleModelHandSegmenterTestCase, self).__init__(methodName)
        self.feature = feature

    def setUp(self):
        pass


    def runTest(self):
        COMPRESSION_WIDTH = 200
        CLASSIFIER = "RF"
        STEP = 2
        POSTPROCESS_PARAMETERS = {
            "sigma" : (3, 3),
            "probabilityThreshold" : 0.3,
            "marginPercentage" : 0.05,
            "compressionWidth" : COMPRESSION_WIDTH,
            "minimumAreaPercentage" : 0.01,
            "maxContours" : 3 }
        postProcessor = PostB2016(**POSTPROCESS_PARAMETERS)

        mask = DATASET_MASKS_PATH.format(dataset, trainingVideo) + trainingMask
        frameFile = DATASET_FRAMES_PATH.format(dataset, trainingVideo) + trainingMask
        maskFrame = Frame.loadMask(mask, COMPRESSION_WIDTH)



        hs = PixelByPixelHandSegmenter(self.feature, COMPRESSION_WIDTH, CLASSIFIER, STEP)
        dataManager = hs.trainClassifier([mask])


        frame = Frame.fromFile(frameFile)
        segment = hs.segmentFrame(frame)
        segment = postProcessor.process(segment)
        postProcessor.visualize(segment)
        cmi, f0, f1 = SegmenterEvaluator.evaluate(maskFrame, segment)


        self.assertGreater(f1, 0.90, msg="Performance has been reduced [F1 Score = {0}]".format(f1))
        self.assertGreater(f0, 0.92, msg="Performance has been reduced [F0 Score = {0}]".format(f0))
        self.assertEqual(numpy.max(segment.matrix), 1, msg="The segmentation result is not binary [1]")
        self.assertEqual(numpy.min(segment.matrix), 0, msg="The segmentation result is no binary [0]")
        self.assertEqual(segment.matrix.shape, (112, 200, 1), msg="The size of the segmentation frame does not match")



class MultiModelHandSegmenterTestCase(unittest.TestCase):

    def __init__(self,methodName="runTest",feature=None):
        super(MultiModelHandSegmenterTestCase, self).__init__(methodName)
        self.feature = feature

    def setUp(self):
        pass

    def runTest(self):
        COMPRESSION_WIDTH = 200
        CLASSIFIER = "RF"
        STEP = 2
        POSTPROCESS_PARAMETERS = {
            "sigma" : (3, 3),
            "probabilityThreshold" : 0.2,
            "marginPercentage" : 0.05,
            "compressionWidth" : COMPRESSION_WIDTH,
            "minimumAreaPercentage" : 0.01,
            "maxContours" : 3 }
        postProcessor = PostB2016(**POSTPROCESS_PARAMETERS)

        gtfile = DATASET_MULTIHSDATAMANAGER_GT_PATH.format(dataset, trainingVideo, trainingMask, self.feature)


        mask = DATASET_MASKS_PATH.format(dataset, trainingVideo)
        frameFile = DATASET_FRAMES_PATH.format(dataset, trainingVideo)
        allMasks = [mask + x for x in os.listdir(mask)]
        allFrames = [frameFile + x for x in os.listdir(mask)]
        nModels = 10
        nAverage = 5

        modelRecommender = NearestNeighbors(nAverage, 200, "HSV-HIST")
        modelRecommender.train(allFrames[:nModels])

        hs = PixelByPixelMultiHandSegmenter(feature, 200, "RF", STEP)
        hs.trainClassifier(allMasks[:nModels], modelRecommender)


        mask = DATASET_MASKS_PATH.format(dataset, trainingVideo) + trainingMask
        maskFrame = Frame.loadMask(mask, COMPRESSION_WIDTH)
        frameFile = DATASET_FRAMES_PATH.format(dataset, trainingVideo) + trainingMask
        frame = Frame.fromFile(frameFile)


        segment = hs.segmentFrame(frame)
        segment = postProcessor.process(segment)
        postProcessor.visualize(segment)
        cmi, f0, f1 = SegmenterEvaluator.evaluate(maskFrame, segment)

        self.assertGreater(f1, 0.88, msg="Performance has been reduced [F1 Score = {0}]".format(f1))
        self.assertGreater(f0, 0.92, msg="Performance has been reduced [F0 Score = {0}]".format(f0))
        self.assertEqual(numpy.max(segment.matrix), 1, msg="The segmentation result is not binary [1]")
        self.assertEqual(numpy.min(segment.matrix), 0, msg="The segmentation result is no binary [0]")
        self.assertEqual(segment.matrix.shape, (112, 200, 1), msg="The size of the segmentation frame does not match")       

        



def load_tests(loader, tests, pattern):
    from egovision.test_parameters import testingModules

    suite = unittest.TestSuite()
    
    if "handSegmentation" in testingModules:
        suite.addTest(MaskBasedDataManagerTestCase('runTest',feature))

        suite.addTest(SingleModelHandSegmenterTestCase('runTest',feature))

        suite.addTest(MultiModelHandSegmenterTestCase('runTest',feature))
            
    return suite

if __name__ == "__main__":
    unittest.main()
