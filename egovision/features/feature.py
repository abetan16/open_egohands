__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['Feature']

import numpy
import cPickle

class Feature(numpy.ndarray):
    """

    Feature wich encapsulates the values of a particular feature in a
    particular frame. It could be created using the feature vector or using a
    filename with the pickle file
    
    This oject inherit the the attributes and methods of a numpy.darray with
    the extra save and load functionality.

    """
    def __new__(cls, input_array, info=None,filename=None):
        if filename == None:
            obj = numpy.asarray(input_array).view(cls)
            obj.info = info
        else:
            obj = pickleLoad(filename)
        return obj
