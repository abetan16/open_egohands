import numpy as np

def scaleEllipse(frame, segment, ellipse):
    scalingFactor = frame.matrix.shape[1]/float(segment.matrix.shape[1])
    scaledCenter = tuple(np.array(ellipse[0])*scalingFactor)
    scaledWidth = tuple(np.array(ellipse[1])*scalingFactor)
    angle = ellipse[2]

    ellipseScaled = (scaledCenter, scaledWidth, angle)
    return ellipseScaled
