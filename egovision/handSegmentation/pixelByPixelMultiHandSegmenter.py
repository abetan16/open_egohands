"""This level is based on a multi-model version of the pixel-by-pixel binary
hand-segmenter proposed by :cite:`Li2013b`. The next figure summarizes the
general idea of the multi-model approach. The grey blocks correspond to the
training while the white blocks to the testing. This multimodel approach is a
strategy to increase the robustnes to illumination changes.

.. image:: ../_images/diagrams/multimodel.png
    :align: center

The first column of the figure contains the manually labelled masks and their
corresponding raw frames. The masks were extracted using the graph cut manual
segmenter provided by :cite:`Li2013b`. Let us denote :math:`N` as the number of
manual labels available in the dataset, and $n$ as the number of training pairs
selected to build a multi-model binary hand-segmenter. For each training pair
:math:`i=1\dots n` a trained binary random forest (:math:`RF_i`) and its global
feature (:math:`GF_i`) are obtained and stored in a pool of illumination models
(second column of the figure).  Each :math:`RF_i` is trained using as features
the LAB values of each pixel in the frame :math:`i` and as class their
corresponding values in the binary masks. As global feature (:math:`GF_i`) we
use the flatten HSV histogram. The choice of the color spaces is based on the
results reported by :cite:`Li2013b` and :cite:`Morerio2013`. Once the
illumination models are trained, a K-Closest-Neighbours structure, denoted as
:math:`K_{RF}`, is estimated using as input the global features :math:`GF_i`.

In the testing phase, the :math:`K_{RF}`: is used as a recommender system
which, given the global features of a new frame, provides the indexes of the
closest :math:`K` illumination models (math:`RF^t`). These models are
subsequently used to obtain :math:`K` possible segmentations (:math:`S^t`),
which are finally fused to obtain the final binary hand-segmentation
(:math:`HS^t`). This procedure is illustrated in the third column of Figure.
Formally, lets denote the testing frame as :math:`t` and its HSV-histogram as
:math:`GF^{t}`. The following equations shows the indexes of the closest
:math:`K` illumination models ordered from closest to furthest based on the
euclidean distance, their corresponding :math:`K` random forest, and their
pixel-by-pixel segmentation applied to :math:`t`.

.. math::
   :nowrap:

   \\begin{eqnarray}
       \Psi^{t} & = & K_{RF}(GF^t|K) \\\\ 
                & = & \{\psi_1^t,\dots,\psi_K^t\} \\\\
       RF^t & = & \{RF_{\psi_1^t},\dots,RF_{\psi_K^t}\} \label{eq:models} \\\\
       S^t & = & \{RF_{\psi_1^t}(t),\dots,RF_{\psi_K^t}(t)\} \\\\
                & = &  \{S_1^t,\dots,S_K^t\} 
   \\end{eqnarray}

The binary hand-segmentation of the frame is the normalized weighted average of
the individual segmentations in :math:`S^t`, which is formally given by nex
equation. Where :math:`\lambda` is a decaying weight factor, selected as
:math:`0.9` based on the results of :cite:`Li2013b`. The weights :math:`S^t`
are then set as :math:`\{0.9,0.9^2,0.9^3 \cdots 0.9^K\} = \{0.9,0.81,0.729
\cdots 0.9^K\}`.  With this in mind the hand-segmentation has 2 parameters to
be defined, namely the number of illumination models (:math:`n`) and the number
of closest random forest to average (:math:`K`). These parameters are dataset
dependent, according to our experiments for the kitchen dataset n=20 and K=5 is
the best performed combination. 

.. math::
   :nowrap:

        \\begin{eqnarray}
            HS^t & = & \\frac{\sum_{j=1}^{K}\lambda^{j}\cdot S^t_j}{\sum_{j=1}^{K}\lambda^{j}} \\
                  & = & \\frac{\sum_{j=1}^{K}0.9^{j}\cdot S^t_j}{\sum_{j=1}^{K}0.9^{j}}
        \\end{eqnarray}


.. automethod:: egovision.handSegmentation.PixelByPixelMultiHandSegmenter.trainClassifier

.. automethod:: egovision.handSegmentation.PixelByPixelMultiHandSegmenter.segmentFrame
"""
__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['PixelByPixelMultiHandSegmenter']

import sys, os
sys.path.append(os.path.abspath('../../'))
from egovision import Frame
import numpy as np
from pixelByPixelHandSegmenter import PixelByPixelHandSegmenter
from handSegmenter import HandSegmenter
from egovision.extras import Sampler
from dataManager import MaskBasedDataManager

class PixelByPixelMultiHandSegmenter(HandSegmenter):
    """
    
    """    

    def __init__(self, feature, compressionWidth, classifier, step=3):
        HandSegmenter.__init__(self, feature, compressionWidth, classifier, step)
        if classifier == "RF_GPU" or classifier == "RF_GPU2":
            self.GPU = True
        self.segmentersList = []
    
    def __getClassifier__(self, position):
        return self.segmentersList[position].classifier
        

    def trainClassifier(self, maskList, modelRecommender=None):

        """ This method train a multimodel pixel by pixel hand segmenter using
        the provided masksFiles and a pretrained ModelRecommender object. The
        trained models are kept as attributes inside the handSegmenter and the
        datamanagers are returned.
        
        :ivar List maskFiles: Filenames to train multiple hand segmenters
        
        :returns: [DataManagers] List of data managers used to train the \
        multiple segmenters.

        Example 1: Training a multimodel hand segmenter::
            
            from egovision.handSegmentation import PixelByPixelMultiHandSegmenter
            from egovision.extras import ObjectPickler
            from egovision.values.paths import DATASET_MASKS_PATH
            from egovision.values.paths import DATASET_FRAMES_PATH
            from egovision.modelRecommenders import NearestNeighbors
            import os

            # DEFINING THE PATHS
            dataset = "GTEA"
            trainingVideo = "GTEA_S1_Coffee_C1"
            maskFolder = DATASET_MASKS_PATH.format(dataset, trainingVideo)
            frameFolder = DATASET_FRAMES_PATH.format(dataset, trainingVideo)
            allMasks = [maskFolder + x for x in os.listdir(maskFolder)]
            allFrames = [frameFolder + x for x in os.listdir(maskFolder)]

            # DEFINING THE PARAMETERS
            nModels = 20
            nAverage = 5
            compressionWidth = 200
            globalFeature = "HSV-HIST" # Histogram of HSV
            classifier = "RF" # Randon Forest
            feature = "LAB"


            # TRAINING THE MODEL RECOMMENDER
            modelRecommender = NearestNeighbors(nAverage, compressionWidth, globalFeature)
            modelRecommender.train(allFrames[:nModels])


            # TRAINING THE MULTIMODEL HAND SEGMENTER
            hs = PixelByPixelMultiHandSegmenter(feature, compressionWidth, classifier)
            hs.trainClassifier(allMasks[:nModels], modelRecommender)


            # SAVING THE HAND SEGMENTER
            ObjectPickler.save(hs, "multiHandSegmenter_test.pk")
        
        Example 2: Loading the previously saved hand-segmenter::

            from egovision.handSegmentation import PixelByPixelMultiHandSegmenter
            from egovision.extras import ObjectPickler

            # LOADING THE HAND-SEGMENTER
            hs = ObjectPickler.load(PixelByPixelMultiHandSegmenter,"multiHandSegmenter_test.pk")

        """
        self.modelRecommender = modelRecommender
        dataManagers = []
        for i in maskList:
            model = PixelByPixelHandSegmenter(self.feature, None, self.classifierType, self.step)
            dm = model.trainClassifier([i])

            self.segmentersList.append(model)
            self.featureLength = dm[0].attributes.shape[1]
            dataManagers.append(dm[0])
        return dataManagers
    
    # @profile
    def segmentFrame(self, frame):

        """ Once the model is trained the next step is to use on new frames.
        This method applies the multi model pixel by pixel hand segmenter on a
        particular frame object.  

        :ivar Frame frame: frame to be segmented
        
        :returns: [Frame] probability map with the skin-like likehood. The \
        returned frame has the compressionWidth to simplify its use on other \
        parts of this library. Be careful this is not the binary mask a \
        continous probability map.

        Example 1: Loading and using a previously saved hand segmenter::

            from egovision import Frame
            from egovision.handSegmentation import PixelByPixelMultiHandSegmenter
            from egovision.extras import ObjectPickler


            # Just one mask and one frame
            datasetFolder = "egovision/dataExamples/GTEA/"
            frameFile = "".join([datasetFolder,"img/GTEA_S1_Coffee_C1/00000780.jpg"])


            # LOADING THE HAND-SEGMENTER
            hs = ObjectPickler.load(PixelByPixelMultiHandSegmenter,"multiHandSegmenter_test.pk")


            # SEGMENTING A NEW FRAME
            frame = Frame.fromFile(frameFile)
            segment = hs.segmentFrame(frame)            

        """
        # INICILIZING RESULTS
        norm = 0


        # THIS PART IS TO AVOID ACCESSING THE ARRAY PER MODEL 
        distances, modelIndexes = self.modelRecommender.predict(frame)
        success, featureVideo = self.featureController.getFeatures(frame)
        desc = featureVideo.next()
        
        if not self.sampler:
            self.sampler = Sampler(self.featureController.height, self.featureController.width, self.step)
        
        # Sampling the descriptor
        descSampled = self.sampler.sampleDescriptor(desc, 3)
        resultSampled = np.zeros(len(descSampled))
        result = np.zeros(len(desc)/3)
        
        descSampled = self.preProcessData(descSampled)

        # Predicting
        resultSampled = None
        for i, modelIndex in enumerate(modelIndexes[0]):
            weight = 0.9**i
            if self.GPU:
                segment = self.segmentersList[modelIndex].classifier.predict(descSampled, weight)
            else:
                segment = self.segmentersList[modelIndex].classifier.predict(descSampled)*(0.9**i)
            norm += weight
            if resultSampled is None:
                resultSampled = segment # THIS IS WHERE g(c|g) COULD BE INCLUDED
            else:
                resultSampled += segment # THIS IS WHERE g(c|g) COULD BE INCLUDED
        
        # Averaging
        resultSampled = resultSampled/float(norm)
        resultSampled = self.postProcessPrediction(resultSampled)
        
        # Reshaping
        result[self.sampler.indexes] = resultSampled
        result = result.reshape((self.featureController.height,
                        self.featureController.width,1))
        result = Frame(result)
        return result
