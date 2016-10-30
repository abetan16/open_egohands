__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['MaskBasedDataManager']

import sys,os
sys.path.append(os.path.abspath('../'))
from egovision.extras import fullVideoName

class MaskBasedDataManager():

    """ The MaskBasedDataManager is an object to store the training features
    (Color Components per pixel) and its corresponding binary ground truths
    (pixels of the masks).  The MaskBasedDataManager can be created from a
    single or a list of frame-mask files. The following image shows some
    examples of the image and binary masks.

    .. image:: ../_images/diagrams/maskExamples.png
        :align: center

    * Example 1: Reading it from the dataset folder and saving it as a pickle::

            from egovision import Frame
            from egovision.handSegmentation import MaskBasedDataManager
            from egovision.handSegmentation import PixelByPixelHandSegmenter
            from egovision.extras import ObjectPickler

            datasetFolder = "egovision/dataExamples/GTEA/"
            mask = "".join([datasetFolder,"masks/GTEA_S1_Coffee_C1/00000780.jpg"])
            img = "".join([datasetFolder,"img/GTEA_S1_Coffee_C1/00000780.jpg"])

            dm = MaskBasedDataManager()
            dm.readDataset([mask],200,"LAB")
    """

    def __init__(self):
        self.LOG_TAG = "[MaskBasedDataManager] "
        self.attributes = []
        self.categories = []
        self.maskFiles = None
        self.feature = None
 

    def readDataset(self,maskFiles, compressionWidth, feature):
        
        """ This method reads a frame and its matching binary mask and store
        the features to be used for training purposes. To manually create the
        binary masks we strongly recommend to use the code of :cite:`Li2003a`,
        or use public available datasets, such as the GTEA :cite:`Fathi2011` or
        the Zombie :cite:`Li2003a` dataset.
        

        * Example 1: Reading the files and saving the DataManager as a pickle::

                from egovision import Frame
                from egovision.handSegmentation import MaskBasedDataManager
                from egovision.handSegmentation import PixelByPixelHandSegmenter
                from egovision.extras import ObjectPickler

                datasetFolder = "egovision/dataExamples/GTEA/"
                mask = "".join([datasetFolder,"masks/GTEA_S1_Coffee_C1/00000780.jpg"])
                img = "".join([datasetFolder,"img/GTEA_S1_Coffee_C1/00000780.jpg"])

                #  READING THE TRAINING FRAMES
                dm = MaskBasedDataManager()
                dm.readDataset([mask],200,"LAB")

                frame = Frame.fromFile(img)

                handSegmenter = PixelByPixelHandSegmenter("LAB", 200, "RF", (3,3), 3)
                handSegmenter.trainClassifier(dm)
                segment = handSegmenter.segmentFrame(frame)

                ObjectPickler.save(dm, "MaskBasedDataManager_test.pk")
                ObjectPickler.save(handSegmenter, "HandSegmenter_test.pk")
        """
        from egovision import Video, Frame
        from egovision.features import FeatureController
        import numpy
        import os
        self.attributes = [[0,0,0]]
        self.categories = []
        maskFiles.sort(key=lambda x: int(os.path.split(x)[1][:-4]))
        self.maskFiles = maskFiles
        self.frameFiles = [x.replace("masks","img")[:-4] + ".bmp" for x in self.maskFiles]
        self.feature = feature
        featureController = FeatureController(compressionWidth, feature)
        self.attributes = numpy.array([[0,0,0]])
        self.categories = numpy.array([])
        for nm, mask in enumerate(maskFiles):
            frame = Frame.fromFile(self.frameFiles[nm],compressionWidth=compressionWidth)
            maskFrame = Frame.loadMask(mask)
            maskFrame = maskFrame.resizeByWidth(frame.matrix.shape[1])
            success, videoFeature = featureController.getFeatures(frame)
            desc = videoFeature.next()
            try:
                desc = desc.reshape((maskFrame.matrix.size,3))
            except:
                import pdb
                pdb.set_trace()
            self.attributes = numpy.vstack((self.attributes,desc))
            self.categories = numpy.hstack((self.categories,maskFrame.matrix.flatten()))
        self.attributes = numpy.delete(self.attributes,0,0)
