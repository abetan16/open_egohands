import unittest
from egovision import Video, Frame
from egovision.values.paths import VIDEO_EXAMPLE_PATH
from egovision.extras import ObjectPickler

class VideoTestCase(unittest.TestCase):

    def __init__(self,methodName="runTest",videoname=None):
        super(VideoTestCase, self).__init__(methodName)
        self.videoname = videoname

    def setUp(self):
        pass

    def videoRead(self):
        videoname = VIDEO_EXAMPLE_PATH.format(self.videoname)
        video = Video(videoname)
        success, frame = video.read()
        video.release()
        # Test reading
        self.assertTrue(success,msg="Video is not reading")
        # Test frame type
        self.assertIsInstance(frame, Frame,msg="Reader does not return Frame")

    def frameSize(self):
        videoName = VIDEO_EXAMPLE_PATH.format(self.videoname)
        video = Video(videoName)
        success, frame = video.read()
        video.release()
        # Test frame width
        self.assertEqual(frame.matrix.shape, (720, 1280,3),
                         msg = "Dimensions of the frame does not match")

    def frameExportImport(self):
        videoname = VIDEO_EXAMPLE_PATH.format(self.videoname)
        video = Video(videoname)
        success, frame = video.read()
        video.release()
        filename = VIDEO_EXAMPLE_PATH.format("frameMatrix.pk")
        ObjectPickler.save(frame, filename)
        frame2 = ObjectPickler.load(Frame, filename)
        self.assertEqual(frame,frame2,msg="Readed frame different of ground truth")

def load_tests(loader,tests,pattern):
    print "creating suit"
    fast = True
    suite = unittest.TestSuite()
    suite.addTest(VideoTestCase('videoRead','BENCHHANDS1.MP4'))
    suite.addTest(VideoTestCase('frameSize','BENCHHANDS1.MP4'))
    suite.addTest(VideoTestCase('frameExportImport','BENCHHANDS1.MP4'))
    return suite

if __name__ == "__main__":
    unittest.main()

