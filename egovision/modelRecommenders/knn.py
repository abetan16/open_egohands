__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['NearestNeighbors']

from egovision.features import FeatureController
from egovision import Frame


class NearestNeighbors:
    def __init__(self, n, compressionWidth, feature):
        from sklearn.neighbors import NearestNeighbors
        self.nneighs = n
        self.recomendationSystem = NearestNeighbors(n_neighbors=n, algorithm='ball_tree')
        self.featureController = FeatureController(compressionWidth, feature)
        self.featureList = []

    def train(self, frameFiles):
        for frameFile in frameFiles:
            frame = Frame.fromFile(frameFile)
            try:
                success, featureVideo = self.featureController.getFeatures(frame)
            except:
                print frameFile
            self.featureList.append(featureVideo.next())
        self.recomendationSystem.fit(self.featureList)

    def kneighbors(self, data):
        return self.recomendationSystem.kneighbors(data)
    
    def predict(self, frame):
        success, data = self.featureController.getFeatures(frame)
        desc = data.next()
        return self.kneighbors(desc.reshape(1,-1))

