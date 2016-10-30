from dataAssociation import PDA
import itertools
import numpy as np

class JPDA(PDA):
    
    def __init__(self):
        PDA.__init__(self)
        self.C = 1e-30
        self.frameNumber = 0

    def mDistances2clutters(self, mDistances):
        usageMatrix = np.array(mDistances < self.getThreshold())
        measurementUsage = np.sum(usageMatrix,1)

        trackUsage = np.sum(usageMatrix,0)
        return measurementUsage, trackUsage, usageMatrix

    def getEvents(self, measurements):
        M = range(len(measurements))
        T = range(1, len(self.trackers) + 1)
        events = []
        mDistances = np.zeros((len(measurements),len(self.trackers)))
        volumes = []
        for i in range(0, len(self.trackers)+1): # trackers from 1 to n included
            dim_y = len(self.trackers[i-1].H)
            if i == 0:
                eventMatrix = np.zeros((len(measurements),len(self.trackers) + 1))
                eventMatrix[:,0] = 1
                events.append(eventMatrix)
            else:    
                volumes.append((2*np.pi)**(dim_y*0.5)*np.linalg.norm(self.trackers[i - 1].s_u)**0.5)
                prediction = self.trackers[i - 1].z_p
                invCovariance = self.trackers[i - 1].s_u_inv
                mDistances[:, i - 1] = self.estimateMahalanobisDistances(prediction,
                                                                 invCovariance,
                                                                  measurements)
                if i <= len(measurements):
                    for measurement in itertools.combinations(M,i):
                        for tracker  in itertools.permutations(T,i):
                            event = [measurement,tracker]
                            eventMatrix = np.zeros((len(measurements),len(self.trackers) + 1))
                            eventMatrix[event] = 1
                            eventMatrix[:,0] = 1
                            eventMatrix[measurement,0] = 0
                            events.append(eventMatrix)
        
        return events, mDistances, volumes

    def getAssignationLikehood(self, measurements):
        from egovision.handTracking.utils import ellipticFormat
        idLikehood = np.zeros((len(measurements),len(self.trackers)))
        for nm, measurement in enumerate(measurements):
            for nt, tracker in enumerate(self.trackers):
                trackerEllipse = ellipticFormat(tracker.x_u)
                trackerL = tracker.idModel.__ellipse2probabilities__([trackerEllipse])[0]
                measurementEllipse = ellipticFormat(measurement)
                measurementL = tracker.idModel.__ellipse2probabilities__([measurementEllipse])[0]
                sameModelL = trackerL[0]*measurementL[0] + trackerL[1]*measurementL[1]
                diffModelL = trackerL[1]*measurementL[0] + trackerL[0]*measurementL[1]
                modelLikehood = sameModelL/float(diffModelL)
                modelLikehood = sameModelL
                idLikehood[nm][nt] = modelLikehood
        return idLikehood

    def getAssignation(self, measurements):
        def evaluateEvent(event, idLikehoods):
            tIndex = np.sum(event[:,1:],0)
            falseM = np.sum(event[:,0])

            t1 = 1
            t2 = 1
            t3 = 1
            for nm, row in enumerate(event):
                for nt, col in enumerate(row):
                    if nt > 0:
                        if event[nm, nt] == 1: # Assigned Measurement and tracker
                            idt = idLikehoods[nm][nt-1]
                            t1 *= idt*expDistances[nm,nt-1]/float(self.C*volumes[nt - 1])
                            t2 *= self.trackers[nt-1].detectionProbability
                            assignation2event.setdefault((nm, nt-1),[]).append(ne)

            
            for nt, usedTracker in enumerate(tIndex):
                if usedTracker == 0:
                    t3 *= (1 - self.trackers[nt].detectionProbability)
            #print falseM, t1, t2, t3
            # t0 = self.C**(falseM)
            t0 = 1
            probability = t0*t1*t2*t3
            
            return probability, [t0,t1,t2,t3]

        events, mDistances, volumes = self.getEvents(measurements)
        idLikehoods = self.getAssignationLikehood(measurements)

        measurementUsage, trackUsage, usageMatrix = self.mDistances2clutters(mDistances)

        expDistances = np.exp(-0.5*mDistances)

        eventProbability = []

        normFactor = 0
        probabilities = []
        assignation2event = {}
        for ne, event in enumerate(events):
            probability, t = evaluateEvent(event, idLikehoods)
            normFactor += probability
            probabilities.append(probability)

        if normFactor > 0:
            probabilities = np.array(probabilities)/normFactor
        else:
            print "norm factor is 0"

        weights = np.zeros((len(measurements), len(self.trackers)))
        for nm, row in enumerate(weights):
            for nt, col in enumerate(row):
                eventList = assignation2event[(nm, nt)]
                weight = 0
                for e in eventList:
                    weight += probabilities[e]

                weights[nm, nt] = weight

        
        b0 = 1 - np.sum(weights,0)
        return events, b0, weights, measurementUsage, trackUsage, usageMatrix

    def nextStep(self, measurements, frameNumber):
        self.frameNumber = frameNumber
        usedMeasurements = []
        self.clutterMeasurements = range(len(measurements))
        trackUsage = [1]*len(self.trackers)
        
        for tracker in self.trackers:
            tracker.predict()
        
        if len(self.trackers) > 0 and len(measurements) > 0:
            assignation, b0, probabilities, measurementUsage, trackUsage, usageMatrix = self.getAssignation(measurements)
            measurementIndexes = range(len(measurements))
            self.clutterMeasurements = filter(lambda x: measurementUsage[x] == 0, measurementIndexes)
            
            # print "----------------------------"
            # print frameNumber
            # print len(measurements), len(self.trackers)
            # print probabilities
            # print measurementUsage
            # print trackUsage
    
        # if self.frameNumber == 660 and self.frameNumber % 2 == 0:
        #     import pdb
        #     pdb.set_trace()

        for nt, tracker in enumerate(self.trackers):
            
            if len(measurements) > 0:
                m = measurements[usageMatrix[:,nt]]
                p = probabilities[usageMatrix[:,nt],nt]
                p0 = b0[nt]
            else:
                m = p = p0 = np.array([])
                validIds = []
             
            tracker.updateState(m, p, p0)
        
        self.__keepAttributeHistory__("likehood")



            




    def __keepAttributeHistory__(self, attr):
        for t in self.trackers:
            if not hasattr(self, attr + "Hist"):
                setattr(self, attr + "Hist", {})
            getattr(self, attr + "Hist").setdefault(t.trackerId,[]).append(getattr(t, attr))

    def __exportAttributeHistory__(self, attr, filename):
        fout = open(filename, "w")
        attribute = getattr(self, attr + "Hist")
        for trackerId in attribute:
            line = [str(trackerId)] + map(str, attribute[trackerId]) + ["\n"]
            if len(line) > 12:
                fout.write(",".join(line))
        fout.close()

            
