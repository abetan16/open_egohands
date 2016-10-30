import numpy

def getLBP(frame, coordinate,windowSize=1):
    coordinate[0] = max(windowSize,min(frame.matrix.shape[0]-windowSize-1,coordinate[0]))
    coordinate[1] = max(windowSize,min(frame.matrix.shape[1]-windowSize-1,coordinate[1]))
    result = numpy.zeros(9)
    gridTop = coordinate[0]-windowSize
    gridLeft = coordinate[1]-windowSize
    pixelValue = frame.matrix[gridTop+windowSize,gridLeft+windowSize]
    grid = frame.matrix[gridTop:gridTop+1+2*windowSize,gridLeft:gridLeft+1+2*windowSize]
    grid = (grid < pixelValue).astype(int)
    result = grid.flatten()
    return result


