""" 
    

"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['IdentificationModel']

from egovision import Frame
from abstractIdentificationModel import AbstractIdentificationModel
from kernelIdentificationModel import KernelIdentificationModel
from egovision.interfaces import cv2i
import numpy as np
import os

class MaxwellFunction:
    
    def __init__(self, d_x, a_x, d_y, a_y):
        self.a_x = a_x
        self.d_x = d_x
        self.a_y = a_y
        self.d_y = d_y
    
    def __unidimensionalMaxwell__(self, d, a, x):
        """
            (2/pi)^0.5*((x+dx^2)*exp(-1*(x+dx)^2/(2*a^2)))/(a^3)
            (2/pi)^0.5*((x+0.03)^2*exp(-1*(x+0.03)^2/(2*0.22^2)))/(0.22^3)
        """
        if x > d:
            pi = np.pi
            c1 = np.sqrt(2/pi)
            numerator = (x-d)**2*np.exp((-(x-d)**2)/float(2*(a**2)))
            denominator = (a**3)
            return c1*numerator/float(denominator)
        else:
            return 0


    def __call__(self, coord):
        x = coord[0]
        y = coord[1]
        p_x = self.__unidimensionalMaxwell__(self.d_x, self.a_x, x)
        p_y = self.__unidimensionalMaxwell__(self.d_y, self.a_y, y)
        return (p_x*p_y)
        


class MaxwellIdentificationModel(AbstractIdentificationModel):
    """
        Based on the empirical distributions, we propose a mathematical
        formulation to match as similar as possible the observed distribution.
        Interestingly we found two independent Maxwell distributions can easily
        fit the empirical distributions. The reasons behind the choice of the
        Maxwell distribution are two: i) It is positive defined ii) It allows
        to include an asymmetry factor in our formulation.

        The mathematical formulation for the left hand (:math:`p_l`) and the
        right hand (:math:`p_r`) is given by equation   

        .. math::
           :nowrap:

            \\begin{eqnarray}
                p_{l}(x, \\theta|\Theta_l^x,\Theta_l^{\\theta}) &=& p(x|\Theta_l^x)  p(\\theta|\Theta_l^{\\theta}), 
            \\end{eqnarray}
        
        and

        .. math::
           :nowrap:

            \\begin{eqnarray}
                p_{r} (x,\\theta|\Theta_r^x,\Theta_r^{\\theta}) &=&  p(1-x|\Theta_r^x)  p(\pi-\\theta|\Theta_r^{\\theta}), \\\\
           \\end{eqnarray}
        
        respectively, where where $p_x$ is the Maxwell distribution with
        parameters $\Theta = [d,a]$. In general $d$ controls the displacement
        of the distribution (with respect to the origin) and $a$ controls its
        amplitude.
                
        .. math::
           :nowrap:

            \\begin{eqnarray}
                p(x|\Theta) = p(x|d,a) &=& \\sqrt{\\frac{2}{\pi}}\dfrac{(x-d)^2}{a^3} \: e^{-\dfrac{(x-d)^2}{2a^{2}}} 
           \\end{eqnarray}


        As notation, the subindex of :math:`\\Theta` refers to the left
        (:math:`l`) or right (:math:`r`) parameters, and the super-index refer
        to the horizontal distance (:math:`x`) or the anti-clockwise angle
        (:math:`\\theta`). In total the maxwell identificationModel has 8
        parameters summarized in equation in the following equation. 
         
        .. math::
           :nowrap:

            \\begin{eqnarray}
              \\begin{bmatrix}
                  \\Theta_l^x & \Theta_l^{\\theta} \\\\ 
                  \\Theta_r^x & \Theta_r^{\\theta} \\\\ 
              \\end{bmatrix}
                     & = &
              \\begin{bmatrix}
                     d_l^x & a_l^x & d_l^{\\theta} & a_l^{\\theta}  \\\\ 
                     d_r^x & a_r^x & d_r^{\\theta} & a_r^{\\theta}  \\\\
              \\end{bmatrix}
            \\end{eqnarray}


    """
    def __init__(self, compressionWidth = 400):
        AbstractIdentificationModel.__init__(self, compressionWidth)
        pass

    def setParameters(self, parametersMatrix):
        """

        The MaxwellIdentificationModel can be directly defined using the
        parameters to avoid the necesity of Left/Right masks::

        Example 2: Maxwell Hand Identifier based on parameters::

            from egovision.handIdentification import MaxwellIdentificationModel
            
            parameters = [[-0.05357189,0.23923865,-0.63053494,0.94603864],\
                          [-0.0851523, 0.21251223, -0.91460727,1.10119661]]
            
            idModel = MaxwellIdentificationModel()
            idModel.setParameters(parameters)
            idModel.visualize()

        """
        params = np.array(parametersMatrix)
        self.leftFunction = MaxwellFunction(*params[0,:])
        self.rightFunction = MaxwellFunction(*params[1,:])
        

    def fit(self, maskFolders):
        """

        Based on the maxwell distribution it is possible to set the parameter
        :math:`d` and :math:`a` using the following equations, where
        :math:`\\bar{\\mu}` and :math:`\\bar{\\sigma}` are average and variance
        estimation comming from the masks. In this equations :mat:`k` is an
        adjustment factor empirically callibrated as 3.5.
        
        .. math::
           :nowrap:

            \\begin{eqnarray}
                a^2 & = & \\frac{k\\bar{\\sigma}\\pi}{3\\pi-8} \\\\
                d & = & \\bar{\\mu} - 2a\\sqrt{\\frac{2}{\\pi}}
            \\end{eqnarray}

        Using the masks of the subject 1 of the kitchen dataset the
        identification model is then defined by the following parameters. These
        parameters can be potentially used to represent the hand-usage and
        dominance of a particular user.
        
        .. math::
           :nowrap:

            \\begin{eqnarray}
              \\begin{bmatrix}
                  \\Theta_l^x & \Theta_l^{\\theta} \\\\ 
                  \\Theta_r^x & \Theta_r^{\\theta} \\\\ 
              \\end{bmatrix}
                     & = &
              \\begin{bmatrix}
                     d_l^x & a_l^x & d_l^{\\theta} & a_l^{\\theta}  \\\\ 
                     d_r^x & a_r^x & d_r^{\\theta} & a_r^{\\theta}  \\\\
              \\end{bmatrix} \\\\
                     & = &
              \\begin{bmatrix}
                     -0.053 & 0.239 & -0.567 & 0.901 \\\\
                     -0.087 & 0.212 & -0.916 & 1.094 \\\\
              \\end{bmatrix}
            \\end{eqnarray}

        The identitifaction model generates the following probability surface.
        Note the similarity with the kernelIdentificationModel, which validates
        the use of the Maxwell distribution as a parametric version of the 
        identification functions.  

        .. image:: ../_images/handIdentification/MaxwellModel.png
            :align: center

        Example 2: Maxwell Hand Identifier based on data::

            from egovision.handIdentification import MaxwellIdentificationModel
            from egovision.handIdentification import KernelIdentificationModel
            from egovision.values.paths import DATASET_LRMASKS_PATH
            
            DATASET = "GTEA"
            VIDEO_NAMES = ["GTEA_S1_Coffee_C1", "GTEA_S1_CofHoney_C1",\\
                           "GTEA_S1_Hotdog_C1", "GTEA_S1_Pealate_C1",\\
                           "GTEA_S1_Peanut_C1", "GTEA_S1_Tea_C1"] 

            VIDEO_FILES = [DATASET_LRMASKS_PATH.format(DATASET, x) for x in VIDEO_NAMES]
            
            idModel = MaxwellIdentificationModel()
            idModel.fit(VIDEO_FILES)
            idModel.visualize()
            



        """
        km = KernelIdentificationModel(self.compressionWidth, 0.8)
        km.fit(maskFolders)
        self.__estimateMaxwellModel__(km.leftMeasurements, km.rightMeasurements)
        self.leftMeasurements = km.leftMeasurements
        self.rightMeasurements = km.rightMeasurements

    
    def __estimateMaxwellModel__(self, leftMeasurements, rightMeasurements):
        leftMeans = np.mean(leftMeasurements, 0)
        leftVars = np.var(leftMeasurements, 0)*3.5
        rightMeans = np.mean(rightMeasurements, 0)
        rightVars = np.var(rightMeasurements, 0)*3.5
        params = np.zeros((2, 4))
        params[0, 0:2] = self.__maxwellParams__(leftMeans[0], leftVars[0])
        params[0, 2:]  = self.__maxwellParams__(leftMeans[1], leftVars[1])
        params[1, 0:2] = self.__maxwellParams__(rightMeans[0], rightVars[0])
        params[1, 2:]  = self.__maxwellParams__(rightMeans[1], rightVars[1])
        self.leftFunction = MaxwellFunction(*params[0,:])
        self.rightFunction = MaxwellFunction(*params[1,:])
        # print params
       
    def __maxwellParams__(self, mean, var):
        a = (var*np.pi/float(3*np.pi-8))**0.5
        d = mean - 2*a*((2/np.pi)**0.5)
        return d, a

    
