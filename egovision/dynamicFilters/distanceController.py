import numpy as np

class DistanceController:
    
    @classmethod
    def mahalanobis(self, x, cov_inv, y):
        """
            x = numpy.array (1 \\times d)
            y = numpy.array (1 \\times d)
            cov = d \\times d
        """
        vector = np.matrix(y - x)
        distance = vector*cov_inv*vector.T
        return float(distance)

    @classmethod
    def kullbackleibler(self, mu1, s1, mu2, s2, s2inv=None):
        if s2inv is None:
            s2inv = np.linalg.inv(s2)
        
        t1 = (s2inv*s1).trace()
        
        v = np.matrix(mu2-mu1)
        t2 = v*s2inv*v.T

        t3 = k = len(mu1)

        det1 = np.linalg.det(s1)
        det2 = np.linalg.det(s2)
        t4 = np.log(float(det2)/det1)

        dkl = 0.5*(t1 + t2 -t3 + t4)
        return float(dkl)


        
