import unittest
import sys,os
sys.path.append(os.path.abspath('../'))
from egovision import Video
from egovision import Frame
from egovision.features import FeatureController
from egovision.features import FeatureVideo
from egovision.features import Feature
from egovision.values.paths import VIDEO_EXAMPLE_PATH
from egovision.values.paths import GROUNDTRUTH_PATH
from egovision.values.paths import GROUNDTRUTH_FEATURE_PATH
from egovision.values.paths import GROUNDTRUTH_VIDEOFEATURE_PATH
from egovision.extras import ObjectPickler
from test_parameters import createGroundTruths
import numpy


if os.getcwd().endswith("features"):
    VIDEO_EXAMPLE_PATH = "../" + VIDEO_EXAMPLE_PATH
    GROUNDTRUTH_PATH = "../" + GROUNDTRUTH_PATH
    GROUNDTRUTH_FEATURE_PATH = "../" + GROUNDTRUTH_FEATURE_PATH
    GROUNDTRUTH_VIDEOFEATURE_PATH = "../" + GROUNDTRUTH_VIDEOFEATURE_PATH

def writeTestDataTruth(featureName,desc):
    fout = open(GROUNDTRUTH_FEATURE_PATH.format(featureName),"w")
    fout.write("desc = " + str(desc))
    fout.close()

class FeatureCreatorTestCase(unittest.TestCase):

    def __init__(self,methodName="runTest",feature=None):
        super(FeatureCreatorTestCase, self).__init__(methodName)
        self.feature = feature

    def setUp(self):
        self.featureController = FeatureController(180, self.feature)

    def runTest(self):
        frame = ObjectPickler.load(Frame, VIDEO_EXAMPLE_PATH.format("frameMatrix.pk"))
        success, desc = self.featureController.getFeatures(frame)
        desc = desc.next()
        if createGroundTruths:
            gtfile = GROUNDTRUTH_FEATURE_PATH.format(self.feature)
            print "[Feature Creator] Ground Truth Created"
            print gtfile
            if not os.path.exists(os.path.split(gtfile)[0]):
                os.makedirs(os.path.split(gtfile)[0])
            ObjectPickler.save(desc, gtfile)
        self.assertIsInstance(desc, numpy.ndarray)
        desc2 = ObjectPickler.load(Feature, GROUNDTRUTH_FEATURE_PATH.format(self.feature))
        numpy.testing.assert_equal(desc,desc2)

    def __str__(self):
        return "".join(["Testing Feature Creator: ",
                        self.feature])

class FeatureVideoTestCase(unittest.TestCase):
    def __init__(self,methodName="runTest",videoname=None, feature=None, extension=".MP4"):
        super(FeatureVideoTestCase, self).__init__(methodName)
        self.feature = feature
        self.videoname = videoname
        self.extension = extension
        self.methodName= methodName
    
    def setUp(self):
        self.featureController = FeatureController(180, self.feature)

    def testVideoFeatureCreator(self):
        from datetime import datetime
        outputfile = GROUNDTRUTH_VIDEOFEATURE_PATH.format(self.videoname, self.feature)
        videoname = VIDEO_EXAMPLE_PATH.format("".join([self.videoname,self.extension]))
        video = Video(videoname)
        success, featureVideo = self.featureController.getFeatures(video)
        self.assertTrue(success,
                          msg = "Impossible to process the features")
        self.assertIsInstance(featureVideo.features, numpy.ndarray,
                          msg = "The video reader is not returning an ndarray")
        
        if createGroundTruths:
            print "[Feature Creator] Ground Truth Created"
            print outputfile
            if not os.path.exists(os.path.split(outputfile)[0]):
                os.makedirs(os.path.split(outputfile)[0])
            success = ObjectPickler.save(featureVideo, outputfile)
            self.assertTrue(success,
                              msg = "Impossible to save the features")

    def testVideoFeatureLoader(self):
        pickleName = GROUNDTRUTH_VIDEOFEATURE_PATH.format(self.videoname, self.feature)
        featureVideo = ObjectPickler.load(FeatureVideo, pickleName)
        self.assertIsInstance(featureVideo.features, numpy.ndarray)

    def __str__(self):
        if self.methodName == "testVideoFeatureLoader":
            extra = "Importer"
        else:
            extra = "Creator"
        return "".join(["Testing Feature Video ",
                        extra, ": ",
                        self.feature, " on ",
                        self.videoname])

def load_tests(loader, tests, pattern):
    from egovision.test_parameters import testingModules
    from test_parameters import features
    from test_parameters import videoTest
    from test_parameters import extension
    from test_parameters import fast
    suite = unittest.TestSuite()

    if "features" in testingModules:
        # This check each of the feature vectors agains the GT
        for feature in features:
            suite.addTest(FeatureCreatorTestCase('runTest',feature))

        # This test that the pickle save of the Video Feature works good.
        # This part only runs in fast is False in the test_parameters
        if not fast:
            suite.addTest(FeatureVideoTestCase('testVideoFeatureCreator',videoTest,feature,extension)) 

        # This test if the loader module fo the Video Feature is working good
        for feature in features:    
            suite.addTest(FeatureVideoTestCase('testVideoFeatureLoader',videoTest,feature,extension)) 
    
    return suite

if __name__ == "__main__":
    unittest.main()
