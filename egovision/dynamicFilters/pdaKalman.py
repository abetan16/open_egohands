from kalmanFilter import KalmanFilter
import numpy as np
from scipy.special import gamma
from distanceController import DistanceController

class PDAKalmanFilter(KalmanFilter):
    
    def __init__(self, A=None, H=None, Q=None, R=None, G=None, X_0=None, threshold=100):
        KalmanFilter.__init__(self, A, H, Q, R, G, X_0)
        self.threshold = threshold
        self.sphereVolume = self.__getSphereVolume__(H.shape[0])
        self.__initializeInnovationCovariance__()
        self.age = 1
        self.totalVisibleCount = 0
        self.consecutiveInvisibleCount = 0
        self.detectionProbability = 0.90

    def setDataAssociationModel(self, dataAssociator):
        self.dataAssociator = dataAssociator
        self.dataAssociator.trackers.append(self)

    def __initializeInnovationCovariance__(self):
        self.s_u = np.identity(self.H.shape[0])
        self.s_u_inv = np.linalg.inv(self.s_u)


    def __getSphereVolume__(self, dimension):
        temp = dimension/float(2)
        return (np.pi**(temp))/gamma(temp+1)
    
    def __getInnovation__(self, prediction, measurements, probabilities):
        y_u = 0
        y_i = []
        for nm, measurement in enumerate(measurements):
            y_u_i = np.array(KalmanFilter.__getInnovation__(self, prediction, measurement))
            y_u += probabilities[nm]*y_u_i
            y_i.append(y_u_i)
        return np.matrix(y_u).T, np.matrix(y_i).T
    
    def __getPMatrix__(self, k_u, y_u, y_i, probabilities):
        rightTerm = np.dot(y_u,y_u.T)
        p = np.zeros_like(rightTerm)
        for ny in range(y_i.shape[1]):
            v = np.matrix(y_i[:,ny])
            p += np.dot(v,v.T)*probabilities[ny]
        return k_u*p*k_u.T
        
    
    def __getStateCovariance__(self, k_u, p_p, s_u, y_u, y_i, probabilities,p0):
        p = self.__getPMatrix__(k_u, y_u, y_i, probabilities)
        I = np.identity(len(self.A))
        t2 =  k_u*s_u*k_u.T
        p_u = p_p - (1-p0)*t2 + p
        self.p_u = np.matrix(p_u)
        diagonal = self.p_p.diagonal()
        # if diagonal[0,9] > 60:
        #     import pdb
        #     pdb.set_trace()
        return p_u

    def nextStep(self, *args):
        raise AttributeError("This filter has not attribute `nextStep`")

    # def nextStepV2(self, measurements):
    #     x_p, p_p, z_p = KalmanFilter.predict(self)

    #     assignation, probabilities, validIds, clutterIds = self.dataAssociator.getAssignation(measurements)

    #     if len(validIds) > 0:
    #         self.totalVisibleCount += 1
    #         self.consecutiveInvisibleCount = 0
    #         k_u, x_u, p_u, y_u, s_u = self.updateState(x_p, p_p, measurements, probabilities)
    #     else:
    #         self.consecutiveInvisibleCount += 1
    #         self.x_u = x_p
    #         self.p_u = p_p
    #         self.measurement = []
    #         self.probabilities = []
    #     self.age += 1

    #     return self.x_u
        
    def updateState(self, measurements, probabilities, p0):
        x_p = self.x_p
        p_p = self.p_p
        if measurements.size > 0:
            y_u, y_i = self.__getInnovation__(x_p, measurements, probabilities)
            s_u = KalmanFilter.__getInnovationCovariance__(self, p_p)
            k_u = KalmanFilter.__getKalmanGain__(self, s_u, p_p)
            x_u = KalmanFilter.__getUpdatedState__(self, x_p, k_u, y_u)
            p_u = self.__getStateCovariance__(k_u, p_p, s_u, y_u, y_i, probabilities,p0)
        else:
            self.x_u = x_p
            self.p_u = p_p
        self.measurement = measurements
        self.probabilities = probabilities
        
        
        
