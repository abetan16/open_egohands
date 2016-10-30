"""
EgoVision: Library to process Egocentric Videos.
"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"


import numpy
from sklearn.metrics import confusion_matrix

def getConfusionMatrix(groundTruth,decisions, nClasses=2):
    """
    Creates the confusion matrix using the groundTruth and the decisions taken.

    :param numpy.ndarray groundTruth: binary list with the ground truth.

    :param numpy.ndarray decisions: binary list with the decisions.

    The following example shows how to build the confusion matrix of a dynamic \
    hand-detector. The procedure includes some if conditions to save the objects \
    once they are trained to reduce the computational time between multiple executions.::

        # ---------------------- REQUIRED MODULES -----------------
        import os
        from egovision.performance import getConfusionMatrix
        from egovision.performance import readGroundTruth
        from egovision.handDetection import DynamicHandDetector
        from egovision.handDetection import HandDetectionDataManager
        from egovision import Video
        # ---------------------------------------------------------


        # ---------------------- PARAMETERS AND PATHS -----------------
        DATASET = "UNIGE"
        VIDEO = "OFFICETEST"
        DATASET_PATH = "egovision/dataExamples/{0}"
        VIDEO_PATH = DATASET_PATH + "/Videos/{0}_{1}.MP4"
        GROUND_TRUTH_PATH = DATASET_PATH + "/GroundTruth/{0}_{1}.csv"
        RESULT_PATH = "results/" # Results and pickles
        DMR_PATH = RESULT_PATH + "{0}_dm.pk" # Data manager pickle
        DETECTOR_PATH = RESULT_PATH + "{0}_hdd.pk" # Dynamic detector
        CLASSIFIER = "SVM"
        FEATURE = "HOG"
        # -------------------------------------------------------------


        # --------------------- CREATING THE DATAMANAGER----------------
        dmr_path = DMR_PATH.format(DATASET)
        dataset_path = DATASET_PATH.format(DATASET)
        if not os.path.isfile(dmr_path):
            dm = HandDetectionDataManager()
            dm.readDataset(datset_path,FEATURE)
            dm.save(dmr_path)
        else:
            print "loading dm"
            dm = HandDetectionDataManager.load(dmr_path)
        # -----------------------------------------------------------


        # --------------------- TRAINNING THE DETECTOR----------------
        detector_path = DETECTOR_PATH.format(DATASET)
        if os.path.isfile(detector_path):
            hdd = DynamicHandDetector(FEATURE, 200, CLASSIFIER)
            hdd.trainClassifier(dm)
            hdd.save(detector_path)
        else:
            hdd = DynamicHandDetector.load(detector_path)
        # -----------------------------------------------------------


        # -------------DETECTING THE HANDS IN THE VIDEO--------------
        test_video = Video(VIDEO_PATH.format(DATASET,VIDEO))
        detection = hdd.classifyVideo(test_video)
        # -----------------------------------------------------------

        # ----------------------- Confusion Matrix --------------------
        gt = readGroundTruth(GROUND_TRUTH_PATH.format(DATASET,VIDEO))
        cm = getConfusionMatrix(gt,detection)
        print cm
        # -----------------------------

    """
    cat = range(nClasses)
    cm = confusion_matrix(groundTruth,decisions,cat)


    #try:
    #    cm = confusion_matrix(groundTruth,decisions,cat)
    #except:
    #    import pdb
    #    pdb.set_trace()

    return cm,cat
        
