__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['cvellipse2points']

import numpy

def cvEllipse2points(ellipse):
    e = ellipse
    # Calculating the center
    center = map(int,e[0])
    ang = e[2]*2*numpy.pi/float(360)
    # Calculating the two small axis
    aux = [0.5*e[1][0]*numpy.cos(ang),0.5*e[1][0]*numpy.sin(ang)]
    newPoint = numpy.array(e[0]) + numpy.array(aux)
    a11 = map(int,newPoint)
    newPoint = numpy.array(e[0]) - numpy.array(aux)
    a12 = map(int,newPoint)
    # Calculating the two long axis
    ang = ang + 0.5*numpy.pi
    aux = [0.5*e[1][1]*numpy.cos(ang),0.5*e[1][1]*numpy.sin(ang)]
    newPoint = numpy.array(e[0]) + numpy.array(aux)
    a21 = map(int,newPoint)
    newPoint = numpy.array(e[0]) - numpy.array(aux)
    a22 = map(int,newPoint)
    return center, a11, a12, a21, a22
