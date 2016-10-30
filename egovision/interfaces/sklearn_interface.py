__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['SklearnInterface']

import sys,os
sys.path.append(os.path.abspath('../../'))
import sklearn as sk
import numpy as np


class SklearnInterface:
    
    def SCV(self, *args, **kargs):
        from sklearn.SVM import SVC
        return SVC(*args, **kargs)

    def RandomForestRegressor(self, *args, **kargs):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(*args, **kargs)

    def __getattr__(self, name):
        return sk.__getattribute__(name)


sklearni = SklearnInterface()

