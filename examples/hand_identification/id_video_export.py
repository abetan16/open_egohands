import sys,os
sys.path.append(os.path.abspath('.'))
from egovision.values.colors import HAND_COLOR, ELLIPSE_COLOR, ELLIPSE_COLOR_TRACK
from egovision.values.paths import DATASET_MASKS_PATH
from egovision.values.paths import DATASET_FRAMES_PATH
from egovision.values.paths import DATASET_VIDEOS_PATH
from egovision import Frame, Video
from egovision.handSegmentation import PixelByPixelMultiHandSegmenter
from egovision.handSegmentation import PostB2016
from egovision.handIdentification import MaxwellIdentificationModel
from egovision.handIdentification import SuperpixelsOcclusionDetector
from egovision.handSegmentation import SegmentVisualizer
from egovision.output import VideoWriter
from egovision.modelRecommenders import NearestNeighbors
from egovision.interfaces import cv2i
import sys
import numpy as np
import os 

# DEFINING SOME PARAMETERS    
RESULTS_PATH = "results/{0}/Videos/{1}" 
N_MODELS = 20
K_AVERAGE = 10
GLOBAL_FEATURE = "HSV-HIST"
COMPRESSION_WIDTH = 420
#CLASSIFIER = "RF_GPU2"
CLASSIFIER = "RF"
STEP = 2
SIGMA = (9, 9)
FEATURE = "LAB"
DATASET = "GTEA"
TRAININGMASK = "00000360.jpg"
argms = sys.argv
if len(argms) > 1:
    TESTINGVIDEO = argms[1]
    TRAININGVIDEO = argms[1]
else:
    TESTINGVIDEO = "S1_Hotdog_C1"
    TRAININGVIDEO = "S1_Hotdog_C1"
TESTINGMASK = "00006060.jpg"
MARGIN_PERCENTAGE = 0.01;
POSTPROCESS_PARAMETERS = {
    "sigma" : SIGMA,
    "probabilityThreshold" : 0.2,
    "marginPercentage" : MARGIN_PERCENTAGE,
    "compressionWidth" : COMPRESSION_WIDTH,
    "minimumAreaPercentage" : 0.005,
    "maxContours" : 3 }
postProcessor = PostB2016(**POSTPROCESS_PARAMETERS)


# DEFINING THE HAND-ID MODEL
ID_PARAMS = [[-0.05357189, 0.23923865, -0.63053494, 0.94603864], \
             [-0.0851523, 0.21251223, -0.91460727, 1.10119661]]
idModel = MaxwellIdentificationModel(COMPRESSION_WIDTH)
idModel.setParameters(ID_PARAMS)





# LOADING THE TRAINING MASK AND FRAME            
mask = DATASET_MASKS_PATH.format(DATASET, TRAININGVIDEO)
frameFile = DATASET_FRAMES_PATH.format(DATASET, TRAININGVIDEO)
allMasks = [mask + x for x in os.listdir(mask)]
allFrames = [frameFile + x[:-4] + ".bmp" for x in os.listdir(mask)]

# TRAINING THE MODEL RECOMMENDER
modelRecommender = NearestNeighbors(K_AVERAGE, COMPRESSION_WIDTH, GLOBAL_FEATURE)
modelRecommender.train(allFrames[:N_MODELS])


# TRAINING THE MULTIMODEL PIXEL BY PIXEL
hs = PixelByPixelMultiHandSegmenter(FEATURE, COMPRESSION_WIDTH, CLASSIFIER, STEP)
dataManagerList = hs.trainClassifier(allMasks[:N_MODELS], modelRecommender)


# DEFINING THE HAND2HAND OCLUSSION DETECTOR
occDetector = SuperpixelsOcclusionDetector(COMPRESSION_WIDTH)
# occDetector.tuneSuperpixelAlgorithm(SIGMA, dataManagerList, hs)
occDetector.setSuperpixelAlgorithm(nSegments = 250, compactness = 7, sigma = SIGMA) # obtained from the optimization


# LOADING TESTING VIDEO
video = Video(DATASET_VIDEOS_PATH.format(DATASET, TESTINGVIDEO) + ".mp4")
fFrame = 0
video.readFrame(fFrame)
writer = VideoWriter(RESULTS_PATH.format(DATASET, TESTINGVIDEO) + "_segmented.avi")
writer.setParametersFromVideo(video)

# leftState = []
# leftFilter = EllipseTracker(COMPRESSION_WIDTH, 1/60.0, modelId = 0)
# rightFilter = EllipseTracker(COMPRESSION_WIDTH, 1/60.0, modelId = 1)

# PER MASK
fout = open(RESULTS_PATH.format(DATASET, TESTINGVIDEO) + ".csv","w")
for frameNumber, frame in enumerate(video):
    print fFrame + frameNumber
    binaryShape = (frame.matrix.shape[0], frame.matrix.shape[1], 1)

    # SEGMENT
    segment = hs.segmentFrame(frame)

    # POSTPROCESS
    segment = postProcessor.process(segment)

    contours, probabilities, split = occDetector.splitOcclusion(postProcessor.others["contours"], frame, frameNumber) 

    # occDetector.visualize() # THIS LINE IS TO VISUALIZE THE OCCLUDED FRAMES
    occDetector.updateState(frame = frame, frameNumber=frameNumber, occusionState = split) # UPDATE OCC_DETECTOR


    # IDENTITY
    contours, ellipses, probabilities = idModel.__contours2probabilities__(contours)
    identities = np.argmax(probabilities,1)
    
    for nc, contour in enumerate(contours):
        measurement = list(ellipses[nc][0]) + list(ellipses[nc][1]) + [ellipses[nc][2]]

        contourSegment = Frame(cv2i.contours2binary([contour], segment.matrix.shape))
        contourSegment = postProcessor.process(contourSegment, targetShape = binaryShape) # EXTRA POST PROCESS
        if nc < 2:
            line = ",".join(map(str, [frameNumber,nc, occDetector.occlusionState]) + \
                            map(str, measurement) +  map(str,probabilities[nc]))
            occDetector.updateState(mask = contourSegment.resizeByWidth(COMPRESSION_WIDTH), identity = identities[nc]) # UPDATE OCC_DETECTOR
        else:
            line = ",".join(map(str, [frameNumber,nc, occDetector.occlusionState]) + \
                            map(str, measurement) +  map(str,[-1,-1]))
        fout.write(line + "\n")
        frame = SegmentVisualizer.__overlapSegmentation__(frame, contourSegment, HAND_COLOR[identities[nc]])
        # frame = SegmentVisualizer.__overlapEllipse__(frame, segment, ellipses[nc], ELLIPSE_COLOR[identities[nc]])

    occDetector.postIteration() # DECLARE CURRENT ITERATION AS FINISHED
    SegmentVisualizer.showFrameQuick("Identification", frame)
    occDetector.visualize()
    writer.writeFrame(frame)

