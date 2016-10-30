import numpy as np

def getParameters(sps):
    H_dict = {} 
    Q_dict = {} 
    R_dict = {} 
    A_dict = {} 
    G_dict = {}
    H_dict = {}
    T_dict = {}
    fps = sps
    delta = 1/float(fps)
    q = 0.03
    r = 32.54
    t = -0.151
    Q_dict["HOG"] = {}
    R_dict["HOG"] = {}
    A_dict["HOG"] = {}
    G_dict["HOG"] = {}
    H_dict["HOG"] = {}
    T_dict["HOG"] = {}
    Q_dict["HOG"]["SVM"] = q*np.array([[(delta**4)/4.0,(delta**3)/2.0],[(delta**3)/2.0,delta**2]])
    R_dict["HOG"]["SVM"] = np.array([r]).T # Meaasurement covariance 
    A_dict["HOG"]["SVM"] = np.array([[1,delta],[0,1]])
    G_dict["HOG"]["SVM"] = np.array([[delta**2/float(2)],[delta]])
    H_dict["HOG"]["SVM"] = np.array([1,0])
    T_dict["HOG"]["SVM"] = t

    Q_dict["DEFAULT"] = {}
    R_dict["DEFAULT"] = {}
    A_dict["DEFAULT"] = {}
    G_dict["DEFAULT"] = {}
    H_dict["DEFAULT"] = {}
    T_dict["DEFAULT"] = {}
    Q_dict["DEFAULT"]["SVM"] = q*np.array([[(delta**4)/4.0,(delta**3)/2.0],[(delta**3)/2.0,delta**2]])
    R_dict["DEFAULT"]["SVM"] = np.array([r]).T # Meaasurement covariance 
    A_dict["DEFAULT"]["SVM"] = np.array([[1,delta],[0,1]])
    G_dict["DEFAULT"]["SVM"] = np.array([[delta**2/float(2)],[delta]])
    H_dict["DEFAULT"]["SVM"] = np.array([1,0])
    T_dict["DEFAULT"]["SVM"] = 0
    return A_dict, H_dict, Q_dict, R_dict, T_dict
