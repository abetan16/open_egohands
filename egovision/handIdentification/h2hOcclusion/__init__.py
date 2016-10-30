"""

The proposed L/R hand-identification models assumes that the hand-like segments
are not occluded or have been split before. If hand-to-hand occlusions were
ignored the L/R hand-identification model would process a larger hand-like
segment and would assign it completely as left or right. Additionally
non-splitting the occlusions would make the tracking of the hands is
considerably more complex due to frequent flickering in the
hand-identification. To avoid these cases we perform an occlusion detection
step followed by a segmentation split when required. The next figure shows some
examples of hand-to-hand occlusions (first column), the split (second column),
and the final segmentation (third column).

.. image:: ../_images/handIdentification/occlusionExamples.png
    :align: center

The design of the occlusion detection is based on the abstract class
(AbstractOcclussionDetector). The objects inherited from this class are
responsible of storing the previous and current information (frame,
identifiedSegments, etc) and detect/split occlusions. In general the
functionality of occlussionDetectors is encapsulated the method splitOcclusion,
which confirms whether a h2h occlusion occurs and, if detected, split the
occluded segment. The next figure shows the structure and methods in the
occlusion detector. As can be seen the dynamic characteristics of this object
introduces considerable complexity in the system. We would recommend to go
through the example bellow to understand how the object is created and used in
whole system.

.. image:: ../_images/occlusion/occlusionDetector.png
    :align: center

In summary, the current occlusionDetector relies on the standard Slic algorithm
to capture the edges in the current and previous frame, and then disambiguate
the super-pixels involved in the occlusion with the closest superpixels of the
previous frame. The next figure shows an occluded frame (first Column), the
previous L/R segmentation and its superpixels (second column). The third column
shows the result obtained. It is noteworthy that the split does not totally
follow the superpixels boundaries because it is a combination of geometric
operations and superpixel decisions as explained in the paper.

.. image:: ../_images/occlusion/algorithmVisualize.png
    :align: center


The following example shows the extended version of the L/R segmentation
example presented above. The main difference in this example is that occlusions
are considered::

    from egovision.values.colors import HAND_COLOR
    from egovision.values.paths import DATASET_MASKS_PATH
    from egovision.values.paths import DATASET_FRAMES_PATH
    from egovision.values.paths import DATASET_VIDEOS_PATH
    from egovision import Frame, Video
    from egovision.handSegmentation import PixelByPixelMultiHandSegmenter
    from egovision.handSegmentation import PostB2016
    from egovision.handIdentification import MaxwellIdentificationModel
    from egovision.handIdentification import SuperpixelsOcclusionDetector
    from egovision.handSegmentation import SegmentVisualizer
    from egovision.modelRecommenders import NearestNeighbors
    from egovision.interfaces import cv2i
    import numpy as np
    import os 
    
    # DEFINING SOME PARAMETERS    
    N_MODELS = 20
    K_AVERAGE = 10
    GLOBAL_FEATURE = "HSV-HIST"
    COMPRESSION_WIDTH = 400
    CLASSIFIER = "RF"
    STEP = 2
    SIGMA = (9, 9)
    FEATURE = "LAB"
    DATASET = "GTEA"
    TRAININGMASK = "00000360.jpg"
    TRAININGVIDEO = "GTEA_S1_Coffee_C1"
    TESTINGVIDEO = "GTEA_S1_Tea_C1"
    TESTINGMASK = "00006060.jpg"
    POSTPROCESS_PARAMETERS = {
        "sigma" : SIGMA,
        "probabilityThreshold" : 0.2,
        "marginPercentage" : 0.01,
        "compressionWidth" : COMPRESSION_WIDTH,
        "minimumAreaPercentage" : 0.005,
        "maxContours" : 3 }
    postProcessor = PostB2016(**POSTPROCESS_PARAMETERS)
    
    
    # DEFINING THE HAND-ID MODEL
    ID_PARAMS = [[-0.05357189,0.23923865,-0.63053494,0.94603864], \\
                 [-0.0851523, 0.21251223, -0.91460727,1.10119661]]
    idModel = MaxwellIdentificationModel(COMPRESSION_WIDTH)
    idModel.setParameters(ID_PARAMS)
    
    
    
    
    
    # LOADING THE TRAINING MASK AND FRAME            
    mask = DATASET_MASKS_PATH.format(DATASET, TRAININGVIDEO)
    frameFile = DATASET_FRAMES_PATH.format(DATASET, TRAININGVIDEO)
    allMasks = [mask + x for x in os.listdir(mask)]
    allFrames = [frameFile + x for x in os.listdir(mask)]
    
    
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
    video = Video(DATASET_VIDEOS_PATH.format(DATASET, TESTINGVIDEO) + ".MP4")
    
    # PER MASK
    for frameNumber, frame in enumerate(video):
        binaryShape = (frame.matrix.shape[0], frame.matrix.shape[1], 1)
    
        # SEGMENT
        segment = hs.segmentFrame(frame)
    
        # POSTPROCESS
        segment = postProcessor.process(segment)
        
        # SOLVE OCCLUSION
        contours, probabilities, split = occDetector.splitOcclusion(postProcessor.others["contours"],
                                                                    frame, frameNumber)
        occDetector.visualize() # THIS LINE IS TO VISUALIZE THE OCCLUDED FRAMES
        occDetector.updateState(frame = frame, frameNumber=frameNumber, occusionState = split) # UPDATE OCC_DETECTOR
    
        # IDENTITY
        contours, identities = idModel.identifyContours(contours)
    
        # VISUALIZE
        for nc, contour in enumerate(contours):
            contourSegment = Frame(cv2i.contours2binary([contour], segment.matrix.shape))
            contourSegment = postProcessor.process(contourSegment, targetShape = binaryShape) # EXTRA POST PROCESS
            frame = SegmentVisualizer.__overlapSegmentation__(frame, contourSegment, HAND_COLOR[identities[nc]])
            occDetector.updateState(mask = contourSegment.resizeByWidth(COMPRESSION_WIDTH), identity = identities[nc]) # UPDATE OCC_DETECTOR
    
        SegmentVisualizer.showFrameQuick("Identification", frame)
        occDetector.postIteration() # DECLARE CURRENT ITERATION AS FINISHED


"""

from superpixelOcclusionDetector import SuperpixelsOcclusionDetector
