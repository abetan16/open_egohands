"""

"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"

from abstractOcclusionDetector import AbstractOcclusionDetector
from egovision.superpixels import Superpixels
from egovision.interfaces import cv2i
from egovision import Frame
from egovision.handSegmentation.postProcessor import ProbabilityThresholder
from egovision.interfaces import sklearni
from collections import Counter
import numpy as np


class SuperpixelsOcclusionDetector(AbstractOcclusionDetector):
    
    def __init__(self, compressionWidth):
        
        self.frameNumber = None

        self.updateState(frame = None, frameNumber = None,
                         leftMask = None, rightMask = None,
                         leftArea = None, rightArea = None,
                         occlusionState = None)
        
        self.others = {}

        # Superpixel algorithm
        self.superpixelAlgorithm = None
        self.superpixelFrameNumber = -50
        self.tempSuperpixelAlgorithm = None
        self.tempSuperpixelFrameNumber = -50
        self.slic = None

        # ProbabilityThresholder
        self.thresholder = ProbabilityThresholder(compressionWidth, 0.1)

        self.compressionWidth = compressionWidth




    def __getSlicResult__(self, frame):
        superpixelAlgorithm =  Superpixels(nSegments = self.superpixelAlgorithm.nSegments,
                                         compactness = self.superpixelAlgorithm.compactness,
                                    compressionWidth = self.compressionWidth,
                                               sigma = self.superpixelAlgorithm.sigma)
        frameCompressed = frame.fromBGR2ColorSpace("LAB")
        frameCompressed = frame.resizeByWidth(self.compressionWidth)
        superpixelAlgorithm.slic(frameCompressed)
        return superpixelAlgorithm

    def __updatePreviousSlicResult__(self, frameNumber, maximumAge):
        # 1. RETRIEVING THE PREVIOUS RESULT
        if (frameNumber - self.superpixelFrameNumber) > maximumAge: # IF THE LAST SUPERPIXEL IS OLD
            self.superpixelAlgorithm = self.__getSlicResult__(self.frame)
            self.superpixelFrameNumber = frameNumber


    def __getIntersectionAreas__(self, contours, leftMask, rightMask):
        def intersectMask(contour, mask):
            contourMask = cv2i.contours2binary([contour], (mask.matrix.shape[0],mask.matrix.shape[1],1))
            mask = cv2i.bitwise_and(contourMask, mask.matrix).reshape(mask.matrix.shape[0],mask.matrix.shape[1],1)
            area = np.sum(mask)
            return mask, area
        intAreas = np.zeros((len(contours),2)).tolist()
        intersectionMatrix = []
        # 0 -> No Hands
        # nc*2 + 1 -> Left Intersection nc contour
        # nc*2 + 2 -> Right Intersection nc contour
        if self.leftMask is not None and self.rightMask is not None:
            intersectionMatrix = np.zeros_like(leftMask.matrix) 
            for nc, contour in enumerate(contours):
                lIntersect, intAreas[nc][0] = intersectMask(contour, leftMask)
                rIntersect, intAreas[nc][1] = intersectMask(contour, rightMask)
                intersectionMatrix += lIntersect*(nc*2 + 1)
                intersectionMatrix += rIntersect*(nc*2 + 2)
        return intersectionMatrix, intAreas
        

    def updateState(self, **kwargs):
        if kwargs.has_key("identity"):
            mask = np.sum(kwargs["mask"])
            area = np.sum(mask.matrix)
            if kwargs["identity"] == 0:
                kwargs["leftMask"] = mask
                kwargs["leftArea"] = area
            else:
                kwargs["rightMask"] = mask
                kwargs["rightArea"] = area
            kwargs.pop("identity")
            kwargs.pop("mask")
        
        # Defining as None the masks to avoid errors
        # they has to be updated in the next steps
        if kwargs.has_key("frameNumber"):
            kwargs["leftMask"] = None
            kwargs["leftArea"] = None
            kwargs["rightMask"] = None
            kwargs["rightArea"] = None

        self.__dict__.update(kwargs)

    def postIteration(self):
        self.occlusionState = self.others["split"]
        if self.tempSuperpixelAlgorithm is not None:
            self.superpixelAlgorithm = self.tempSuperpixelAlgorithm
            self.frame = self.tempFrame
            self.superpixelFrameNumber = self.tempFrameNumber

    def tuneSuperpixelAlgorithm(self, sigma, dataManagerList, handSegmenter):
        self.setSuperpixelAlgorithm(100, 10)
        self.superpixelAlgorithm.train(handSegmenter, dataManagerList)
    
    def setSuperpixelAlgorithm(self, nSegments = 100, compactness = 10, sigma = (9,9)):
        self.superpixelAlgorithm = Superpixels(nSegments = nSegments,
                                               compactness = compactness,
                                               compressionWidth = self.compressionWidth,
                                               sigma = sigma)


    def getOcclusionState(self, contours, intersectionMatrix, intersectionAreas):
        state = None
        description = ""
        intersectionMatrix = []
        intAreas = np.zeros((2,2))
        if len(contours) > 0:
            if self.leftArea != 0 and self.leftArea is not None:
                if self.rightArea != 0 and self.rightArea is not None:
                    totalPrevious = self.leftArea + self.rightArea
                    if (0.7*totalPrevious <= np.sum(intersectionAreas[0]) <= 1.3*totalPrevious) and all(np.array(intersectionAreas[0]) > 100):
                        if self.occlusionState:
                            description, occluded = "Occlusion2Occlusion", True
                        else:
                            description, occluded = "Independent2Occlusion", True
                    else:
                        if len(contours) == 1:
                            description, occluded = "One contours no big intersection", False
                        else: # 2 contours
                            description, occluded = "Two contours no big intersection", False
                else:
                    description, occluded = "NoPreviousRight", False
            else:
                description, occluded = "NoPreviousLeft", False
        else:
            description, occluded = "NoHands", False

        return description, occluded
        
    def trainNearestSuperpixelFinder(self):
        nearestSpxData = []
        index2id = {} # Move from superpixelid to hand id
        nnId2spxId = {}
        idMatrix = self.leftMask.matrix + 2*self.rightMask.matrix
        count = 0
        for spxIndex in range(self.superpixelAlgorithm.lastNumberOfPixels):
             y = int(self.superpixelAlgorithm.lastMeans[spxIndex][1])
             x = int(self.superpixelAlgorithm.lastMeans[spxIndex][2])
             if not np.isnan(y):
                 handId = int(idMatrix[y,x])
             else:
                 handId = 0
             index2id[spxIndex] = int(handId)
             if handId != 0:
                 data = self.superpixelAlgorithm.lastMeans[spxIndex][1:]
                 data[2:] = data[2:]/float(self.superpixelAlgorithm.compactness)
                 nearestSpxData.append(data)
                 nnId2spxId[count] = spxIndex
                 count += 1
        
        self.superpixelAlgorithm.nearestSpxFinder = sklearni.neighbors.NearestNeighbors(n_neighbors=min(5,len(nearestSpxData)),algorithm='brute')
        self.superpixelAlgorithm.nearestSpxFinder.fit(nearestSpxData)
        self.superpixelAlgorithm.nnId2spxId = nnId2spxId
        self.superpixelAlgorithm.index2id = index2id    


    def splitOcclusion(self, contours,
                        frame, frameNumber):
        """
            return leftContour, rightContour
        """
        def boundFinalMasks(mask, occludedMask, previousMask, opositePrevious):
            mask = cv2i.bitwise_and(mask, occludedMask)
            intersection = cv2i.bitwise_and(occludedMask, previousMask.matrix.astype(np.uint64))
            mask = cv2i.bitwise_or(mask, intersection)
            intersection = cv2i.bitwise_and(occludedMask, opositePrevious.matrix.astype(np.uint64))
            mask = mask - intersection
            mask = np.reshape(mask,(mask.shape[0],mask.shape[1],1))
            return mask
        # if frameNumber > 1338:
        #     description, occluded, intersectionMatrix, probabilities = self.getOcclusionState(contours)

        # DETECT IF THEY ARE OCCLUDED 
        intersectionMatrix, intersectionAreas = self.__getIntersectionAreas__(contours, self.leftMask, self.rightMask)
        description, occluded = self.getOcclusionState(contours, intersectionMatrix, intersectionAreas)
        # COMMENT: intersectionMatrixis the first criteria to solve the occlusion
        
        # print str(frameNumber) + " " + str(occluded) + ": " + description
        split = False

        if occluded:
            print occluded
            handsFound = {0:False, 1:False}
            
            # 0. UPDATE PREVIOUS SUERPIXELS IF REQUIERED -> self.superpixelAlgorithm
            self.__updatePreviousSlicResult__(frameNumber, 50) # If is old result update the state
            self.trainNearestSuperpixelFinder()
            
            # 1. NEW SUPERPIXELS -> self.tempSuperpixelAlgorithm
            self.tempSuperpixelAlgorithm = self.__getSlicResult__(frame) # The result is kept in the object
            # 2. Initialize empty binaries 
            width = self.leftMask.matrix.shape[1]
            height = self.leftMask.matrix.shape[0]
            binaryMasks = [[np.zeros((height, width, 1),dtype=np.uint8),
                            np.zeros((height, width, 1),dtype=np.uint8)],
                           [np.zeros((height, width, 1),dtype=np.uint8),
                            np.zeros((height, width, 1),dtype=np.uint8)]]
            # 3. Biggest contour to binary
            extraContours = contours[1:]
            occludedMask = cv2i.contours2binary([contours[0]], (height,width,1)).astype(np.uint64)
            # occludedMaskCenters = cv2i.contours2binary([contours[0]], (height,width,3))


            # -------------------------------------
            # AVERAGING THE LAST SUPERPIXELS
            # -------------------------------------
            for spxId in range(self.tempSuperpixelAlgorithm.lastSuperpixelId):
                y = int(self.tempSuperpixelAlgorithm.lastMeans[spxId][1])
                x = int(self.tempSuperpixelAlgorithm.lastMeans[spxId][2])
                if not np.isnan(x):
                    isInside = int(occludedMask[y,x])
                    if isInside == 1:
                        if 0 < intersectionMatrix[y,x] < 2:
                            # cv2i.circle(occludedMaskCenters, (x,y), 4, (0,0,1))
                            intersectionDecision = intersectionMatrix[y,x] - 1
                            handId = int(intersectionDecision)
                            algorithmType = 0
                        else:
                            # cv2i.circle(occludedMaskCenters, (x,y), 4, (0,1,0))
                            data = self.tempSuperpixelAlgorithm.lastMeans[spxId][1:]
                            data[2:] = data[2:]/float(self.tempSuperpixelAlgorithm.compactness)
                            closestPoints = self.superpixelAlgorithm.nearestSpxFinder.kneighbors([data])
                            closestSuperpixels = [self.superpixelAlgorithm.nnId2spxId[int(point)] for point in closestPoints[1][0]]
                            handsIds = [int(self.superpixelAlgorithm.index2id[spx] - 1) for spx in closestSuperpixels]
                            handsIdsCounting = Counter(handsIds)
                            counts = handsIdsCounting.most_common()
                            handId = counts[0][0]
                            algorithmType = 1

                        if handId >= 0:
                            if handId >= 2:
                                try:
                                    handId = counts[1][0]
                                except:
                                    handId = 1
                            handsFound[handId] = True
                            binaryMasks[handId][algorithmType][self.tempSuperpixelAlgorithm.superpixels2pixels[spxId]] = 1
            # TO VISUALIZE
            # --------------------------------
            # probMap = np.hstack((binaryMasks[0].matrix,binaryMasks[1].matrix))
            # probMap = np.multiply(probMap,[255,255,255]).astype(np.uint8)
            # cv2.imshow("prob_map",probMap)
            # for i in range(10):
            #     cv2.waitKey(1)
            
            
        

            leftMask, rightMask = np.sum(binaryMasks,1)
            leftMask = boundFinalMasks(leftMask, occludedMask, self.leftMask, self.rightMask)
            rightMask = boundFinalMasks(rightMask, occludedMask, self.rightMask, self.leftMask)
            self.tempLeftMask = Frame(leftMask)
            self.tempRightMask = Frame(rightMask)
            handsFound[0] = np.any(self.tempLeftMask)
            handsFound[1] = np.any(self.tempRightMask)

            contours = []
            intAreas = []
            if handsFound[0]:
                contours.append(self.thresholder.process(self.tempLeftMask)[0])
            
            if handsFound[1]:
                contours.append(self.thresholder.process(self.tempRightMask)[0])

            if handsFound[0] and handsFound[1]:
                split = True
            

            intersectionAreas.pop(0)
            intersect, intAreas = self.__getIntersectionAreas__(contours, self.leftMask, self.rightMask)
            intersectionAreas = intAreas + intersectionAreas
            self.tempFrame = frame
            self.tempFrameNumber = frameNumber
            self.tempSplit = split
            self.tempOccludedMask = Frame(occludedMask.astype(np.uint8))
            contours.extend(extraContours)
        self.others["contours"] = contours
        self.others["intersectionAreas"] = intersectionAreas
        self.others["split"] = split
        return contours, intersectionAreas, split
