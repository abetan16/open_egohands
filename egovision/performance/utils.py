import numpy as np

def readGroundTruth(filename):
    """
      
    Import a csv file with the the ground truth of a video.

    :param String filename: filename including the extension.

    """
    groundTruth = filename
    finp = open(groundTruth,"r")
    gt = np.array([int(x) for x in finp.readlines()])
    return gt
