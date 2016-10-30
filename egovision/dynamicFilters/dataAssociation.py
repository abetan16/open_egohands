from distanceController import DistanceController
import numpy as np


class PDA:
    
    def __init__(self):
        self.b = 1e-30
        self.trackers = []

    def getPrediction(self):
        return self.trackers[0].z_p

    def getInvCovariance(self):
        return self.trackers[0].s_u_inv

    def getThreshold(self):
        return self.trackers[0].threshold

    def getMeasurementsIds(self, assignation):
        rowSum = np.sum(assignation[:,1:],1)
        measurementsIds = range(assignation.shape[0])
        clutterIds = filter(lambda x: rowSum[x] == 0, measurementsIds)
        validIds = filter(lambda x: rowSum[x] != 0, measurementsIds)

        return clutterIds, validIds

        

    def getOmega(self, measurements):
        assignation = np.zeros([len(measurements), len(self.trackers)+1])
        probabilities = np.zeros([len(measurements), len(self.trackers)])
        b0probs = []
        threshold = self.getThreshold()
        validIds = []
        for nt, tracker in enumerate(self.trackers):
            prediction = tracker.z_p
            invCovariance = tracker.s_u_inv
            mDistances = self.estimateMahalanobisDistances(prediction, invCovariance, measurements)
            trackerValidIds = []
            for nm, measurement in enumerate(measurements):
                if mDistances[nm] < threshold:
                    assignation[nm, nt+1] = 1
                    assignation[nm, 0]  = 0
                    trackerValidIds.append(nm)
            b0, probabilities[trackerValidIds,nt] = self.getAssignationProbabilities(mDistances[trackerValidIds])
            b0probs.append(b0)
            validIds.append(trackerValidIds)
        
        clutterIds, usedIds = self.getMeasurementsIds(assignation)
        
        print validIds
        return assignation, b0probs, probabilities, validIds, clutterIds

    def getAssignation(self, measurements):
        return self.getOmega(measurements)

    def estimateMahalanobisDistances(self, prediction, inverseCovariance, measurements):
        mahalanobisDistances = []
        s_u_inv = inverseCovariance
        z_p = prediction
        for nm, m in enumerate(measurements):
            x = measurements[nm]
            mahalanobisDistances.append(DistanceController.mahalanobis(z_p, s_u_inv, x))
        return np.array(mahalanobisDistances)

    def getAssignationProbabilities(self, mDistances):
        numerators = np.exp(-1*mDistances/2.0)
        denominator = self.b + sum(numerators) 
        probabilities = numerators/float(denominator)
        b0 = self.b/float(denominator)
        return b0, probabilities

    
    def nextStep(self, measurements):
        usedMeasurements = []
        self.clutterMeasurements = range(len(measurements))
        for tracker in self.trackers:
            tracker.predict()
        
        if len(self.trackers) > 0:
            assignation, b0, probabilities, validIds, clutterIds = self.getAssignation(measurements)
            self.clutterMeasurements = clutterIds
    
        for nt, tracker in enumerate(self.trackers):
            
            if len(validIds[nt]) == 0:
                m = np.array([])
                p = np.array([])
                p0 = 0
            else:
                m = measurements[validIds[nt]]
                p = probabilities[validIds[nt],nt]
                p0 = b0[nt]
                

            tracker.updateState(m, p, p0)
            tracker.age += 1
            
            if len(validIds[nt]) > 0:
                tracker.totalVisibleCount += 1
                tracker.consecutiveInvisibleCount = 0
            else:
                tracker.consecutiveInvisibleCount += 1
            tracker.age += 1
        


