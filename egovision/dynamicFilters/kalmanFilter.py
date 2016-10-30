__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['KalmanFilter']

import numpy as np

class KalmanFilter:
    """

    Linear Kalman filter, for constant force model.

    """
    def __init__(self,A=None,H=None,Q=None,R=None, G=None, X_0=None):
        self.A = A
        self.H = H
        self.Q = Q
        self.G = G
        self.R = R
        self.qTerm = np.dot(np.dot(self.G,self.Q),self.G.T)
        self.p_u = np.identity(len(A))
        self.K = None
        self.X_0 = X_0
        self.step = 0
        self.x_u = np.array(X_0)
        self.update = []
        self.innovation = []
        self.measurement = []
    
    def __getInnovation__(self, prediction, measurement):
        y_u = measurement - self.z_p
        self.y_u = y_u
        return y_u

    def __getInnovationCovariance__(self, p_p):
        s_u = self.R + self.H*p_p*self.H.T
        self.s_u = s_u
        return s_u

    def __getKalmanGain__(self,s_u,p_p):
        self.s_u_inv =  np.linalg.inv(s_u)
        k_u = np.dot(np.dot(p_p,self.H.T),self.s_u_inv)
        self.k_u = k_u
        return k_u
     
    def __getUpdatedState__(self, x_p, k_u, y_u):
        x_u = x_p + np.dot(k_u,np.array(y_u.T)[0])
        x_u = np.array(x_u)[0]
        self.x_u = x_u
        return x_u

    def __getStateCovariance__(self, k_u, p_p):
        p_u = np.dot(np.identity(len(self.A)) - np.dot(k_u,self.H),p_p)
        self.p_u = np.matrix(p_u)
        return p_u

    def processModel(self):
        posterior = np.dot(self.A,self.x_u)
        return posterior

    def nextStep(self, measurement):
        x_p, p_p = self.predict()
        k_u, x_u, p_u, y_u, s_u = self.updateState(x_p,p_p,measurement)
        self.step += 1
        return self.x_u
        
    def predict(self):
        x_p = self.processModel()
        p_p = np.matrix(self.A*self.p_u*self.A.T + self.qTerm)
        self.x_p = x_p
        self.p_p = p_p
        self.z_p = z_p = np.dot(self.H, x_p)
        return x_p, p_p, z_p
    
    def updateState(self, x_p, p_p, st):
        y_u = self.__getInnovation__(x_p, st)
        s_u = self.__getInnovationCovariance__(p_p)
        k_u = self.__getKalmanGain__(s_u, p_p)
        x_u = self.__getUpdatedState__(x_p, k_u, y_u)
        p_u = self.__getStateCovariance__(k_u, p_p)
        self.measurement = st
        return k_u, x_u, p_u, y_u, s_u

    
    def fullExecution(self):
        success = True
        states = []
        update = []
        innovation = []
        innovation_s = []
        count = 0
        while success:
            success = self.nextStep()
            if success:
                states.append(float(self.state[0]))
                innovation.append(float(self.innovation[0]))
                innovation_s.append(float(self.innovation_s[0]))
                update.append(float(self.measurement))
            count += 1
        return states, update, innovation, innovation_s

