__author__ = "Alejandro Betancourt"
__credits__ = "Copyright (c) 2015 Alejandro Betancourt"
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Alejandro Betancourt"
__email__ = "abetan16@gmail.com"
__all__ = ['HandDetector','DynamicHandDetector']

import sys, os
sys.path.append(os.path.abspath('../'))
import numpy
import scipy
from egovision import dynamicFilters as df

from sklearn.svm import SVC


class HandDetector:
    """
    
    HandDetector is an object trained to find if the user hands are present in
    a particular frame of a video. This object contains the used
    featureController and the trained classifier to guarantee consistency
    between the training stage and the testing stage. 

    A HandDetector could be initialized from scratch, initializing a new
    featureController as well as a new classifier and scaler. The second option
    is to load an existing handDetector using the method load.

    :param str feature: String representing the feature thas is going to be\
    used.

    :param float compressionWidth: Proportion of the original width of the\
    frames to be used in the resizing stage before estimate the features. If\
    compression rate is 1 then the original image is used [Default = 0.16].


    :param str classifier: String representing the classifier to be used. By\
    now only SVM is implemented to keep the hand detector behaviour stable. 


    :ivar str LOG_TAG: Tag to be used for debuging purposes

    :ivar str feature: String representing the feature that is being used.

    :ivar FeatureController FeatureController: Instance to be used to extract\
    the features from the frames.

    :ivar Classifier classifier: sklearn classifier.

    :ivar Scaler scaler: sklearn scaler.


    * Example 1::
    
        import egovision
        from egovision import Video
        from egovision.handDetection import HandDetectionDataManager, HandDetector
        from egovision.extras import ObjectPickler
        filename = 'test_dm.pk'
        videoname = 'egovision/dataExamples/BENCHTEST.MP4'
        dm = ObjectPickler.load(HandDetectionDataManager, filename)
        hd = HandDetector("HOG", 200, "SVM")
        hd.trainClassifier(dm)
        video = Video(videoname)
        hands = hd.classifyVideo(video,dtype="integer")
        ObjectPickler.save(hd, "test_hd.pk")
        print hands

    * Example 2::
    
        import egovision
        from egovision import Video
        from egovision.handDetection import HandDetector
        from egovision.extras import ObjectPickler
        videoname = 'egovision/dataExamples/BENCHTEST.MP4'
        video = Video(videoname)
        detectorFilename = 'test_hd.pk'
        hd = ObjectPickler.load(HandDetector, detectorFilename)
        hands = hd.classifyVideo(video,dtype="integer")
        print hands

    """
    def __init__(self, feature, compressionWidth, classifier):
        from egovision.features import FeatureController
        self.TAG_LOG = "[Hand Detector] "
        self.feature = feature
        self.featureController = FeatureController(compressionWidth, feature)
        self.featureLength = None
        if classifier == "SVM":
            from sklearn import svm
            self.classifier = svm.SVC(kernel="linear",probability=True)        
        elif classifier == "RF":
            from sklearn import ensemble
            self.classifier = ensemble.RandomForestClassifier()        
        else:
            from exceptions import UnavailableClassifier
            raise UnavailableClassifier("Sorry, {0} is not implemented yet!".format(classifier))
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
    

    def trainClassifier(self,dataManager):
        """

        This method train the classifier using a dataManager Object. It is
        important to consider that the dataManagers already contain the
        estimated features of the possitive and negative samples. It is a good
        practice to verify that the parameters of saved in the datamanager are
        exactly the same as the HandDetector FeatureController.

        :ivar HandDetectionDataManager dataManager: Pre-processed training dataset

        * Example 1::
        
            import egovision
            from egovision import Video
            from egovision.handDetection import HandDetectionDataManager, HandDetector
            from egovision.extras import ObjectPickler
            filename = 'test_dm.pk'
            videoname = 'egovision/dataExamples/BENCHTEST.MP4'
            dm = ObjectPickler.load(HandDetectionDataManager, filename)
            hd = HandDetector("HOG", 200, "SVM")
            hd.trainClassifier(dm)
            video = Video(videoname)
            hands = hd.classifyVideo(video,dtype="integer")
            ObjectPickler.save(hd, "test_hd.pk")
            print hands
        
        """

        if not self.classifier != []:
            from exceptions import InexistentClassifier
            raise InexistentClassifier("Any classifier has been defined")
        elif not self.scaler != []:
            from exceptions import InexistentScaler
            raise InexistentScaler("Any scaler has been defined")
        else:
            self.scaler.fit(dataManager.attributes.astype(numpy.float))
            scaledAttributes = self.scaler.transform(dataManager.attributes.astype(numpy.float))
            self.classifier.fit(scaledAttributes,dataManager.categories)
            self.featureLength = scaledAttributes.shape[1]


    def classifyVideo(self,video,dtype="integer"):
        """

        This method perform a full scan detection over a video. The result is a
        numpy.darray with the detection result. The result could be a binarized
        array (dtype="integer") or a real number(dtype="float") with
        positive(negative) value if the hands are present(absent) in each of
        the frames. 

        :ivar Video video: Video object

        :ivar str dtype: Type of result expected. "float" is only available if\
        the classifier is an SVM.

        * Example 1::
        
            import egovision
            from egovision import Video
            from egovision.handDetection import HandDetectionDataManager, HandDetector
            from egovision.extras import ObjectPickler
            filename = 'test_dm.pk'
            videoname = 'egovision/dataExamples/BENCHTEST.MP4'
            dm = ObjectPickler.load(HandDetectionDataManager, filename)
            hd = HandDetector("HOG", 200, "SVM")
            hd.trainClassifier(dm)
            video = Video(videoname)
            hands = hd.classifyVideo(video,dtype="integer")
            ObjectPickler.save(hd, "test_hd.pk")
            print hands

        """
        result = []
        for nf, frame in enumerate(video):
            hand = self.classifyFrame(frame,dtype)
            result.append(hand[0])
        return numpy.vstack(result)

    def classifyFrame(self,frame,dtype="integer"):
        """

        This method detects the hands in a Frame and return: 1) a binarized
        array (dtype="integer") or, 2) a real number(dtype="float") with
        positive(negative) value if the hands are present(absent) in each of
        the frames. 

        :ivar Frame frame: Frame to be processed

        :ivar str dtype: Type of result expected. "float" is only available if\
        the classifier is an SVM.

        * Example 1::
        
            import egovision
            from egovision import Video
            from egovision.handDetection import HandDetector
            from egovision.extras import ObjectPickler
            videoname = 'egovision/dataExamples/BENCHTEST.MP4'
            video = Video(videoname)
            frame = video.next()
            detectorFilename = 'egovision/dataExamples/UNIGEmin/handDetectors/GroundTruths/RGB_SVM.pk'
            hd = ObjectPickler.load(HandDetector, detectorFilename)
            hands = hd.classifyFrame(frame,dtype="integer")
            print "integer", hands
            hands = hd.classifyFrame(frame,dtype="float")
            print "float", hands

        """
        if self.feature == None or self.featureController.compressionWidth == None:
            from exceptions import UndefinedParameters
            raise UndefinedParameters(self.TAG_LOG + "Parameters are not defined,\
use defineParameters(feature,compressionWith,minWidth)")
        else:

            success, descriptor = self.featureController.getFeatures(frame)
            result = self.classifyFeatureVector(descriptor.next(),dtype)
            return result


    def classifyFeatureVideo(self, featureVideo,dtype="integer"):
        """

        This method detects the hands in a Feature Video and returns: 1) a
        binarized array (dtype="integer") or, 2) a real number(dtype="float")
        with positive(negative) value if the hands are present(absent) in each
        of the frames. 

        The main advantage of this function over classifyFrame is that if the
        experiment is well designed then this method does not requiere to
        estimate the features each time.

        :ivar FeatureVideo featureVideo: Feature object to be processed

        :ivar str dtype: Type of result expected. "float" is only available if\
        the classifier is an SVM.

        * Example 1::
        
            import egovision
            from egovision.handDetection import HandDetector
            from egovision.features import FeatureVideo
            from egovision.extras import ObjectPickler
            videoname = 'egovision/dataExamples/BENCHTEST.MP4'
            detectorFilename = 'egovision/dataExamples/UNIGEmin/handDetectors/GroundTruths/RGB_SVM.pk'
            fvname = 'egovision/dataExamples/GroundTruths/features/BENCHTEST_RGB.pk'
            fv = ObjectPickler.load(FeatureVideo, fvname)
            hd = ObjectPickler.load(HandDetector, detectorFilename)
            hands = hd.classifyFeatureVideo(fv,dtype="integer")
            print hands
            hands = hd.classifyFeatureVideo(fv,dtype="float")
            print hands

        """

        result = []
        for f in featureVideo.features[0:10]:
            hand = self.classifyFeatureVector(f, dtype)
            result.append(hand[0])
        return numpy.array(result)

        

        
    def classifyFeatureVector(self,feature,dtype="integer"):
        """

        This method detects the hands in a Feature vector of a frame and
        returns: 1) a binarized array (dtype="integer") or, 2) a real
        number(dtype="float") with positive(negative) value if the hands are
        present(absent) in each of the frames. 

        The main advantage of this function over classifyFrame is that if the
        experiment is well designed then this method does not requiere to
        estimate the features each time.

        :ivar Feature feature: Feature object to be processed

        :ivar str dtype: Type of result expected. "float" is only available if\
        the classifier is an SVM.

        * Example 1::
        
            import egovision
            from egovision import Video
            from egovision.handDetection import HandDetector
            from egovision.extras import ObjectPickler
            videoname = 'egovision/dataExamples/BENCHTEST.MP4'
            detectorFilename = 'test_hd.pk'
            hd = ObjectPickler.load(HandDetector, detectorFilename)
            video = Video(videoname)
            frame = video.next()
            success, featureVideo = hd.featureController.getFeatures(frame)
            feature = featureVideo.next()
            hands = hd.classifyFeatureVector(feature, dtype="integer")
            print "integer", hands
            hands = hd.classifyFeatureVector(feature, dtype="float")
            print "float", hands            
        """
        try:
            descriptor = self.scaler.transform(feature.astype(numpy.float).reshape(1,-1))
        except ValueError as e:
            from exceptions import SizeError
            raise SizeError(self.TAG_LOG + "The size of the feature vector does not \
match, Verify size of the features")
        if dtype == "integer":
            result = self.classifier.predict(descriptor)
        elif dtype == "float":
            result = numpy.array(self.__floatPredict__(descriptor))
        return result

    def binarizeDetections(self,detections,th=0):
        detections[detections>th] = 1
        detections[detections<=th] = 0
        return detections



    def __floatPredict__(self, descriptor):
        if isinstance(self.classifier,SVC):
            import time
            t0 = time.time()
            res =  scipy.linalg.blas.dgemm(alpha=1.0,a=self.classifier.coef_.T,b=descriptor.T,trans_a=True) + \
                    self.classifier.intercept_
            t1 = time.time()
            #res = numpy.dot(self.classifier.coef_,descriptor.T) + self.classifier.intercept_
            return res[0]
        else:
            res = float(self.classifier.predict(descriptor))
            return res


class DynamicHandDetector(HandDetector):
    """

    DynamicHandDetector is an object almost identical to the nomral
    HandDetector. The difference is mainly that it takes into account the
    temporal dimension, namely the previous decisions taken. To accomplish this
    the DynamicHandDetector mix the frame by frame classifier with a dynamic
    filter. The tunning process of this object is much more complex than the
    atemporal strategy but could lead to substantial improvements in the
    results. The DynamicHandDetector only support SVM classifiers. Regarding
    the features, all the available features could be used, however, only the
    estimated parameters for the HOG are provided. These parameter were found
    in the paper []. If you are interested in the use of the other features we
    suggest you to follow a similar procedure to the one presented in our
    paper.


    :param str feature: String representing the feature thas is going to be\
    used.

    :param float compressionRate: Proportion of the original width of the\
    frames to be used in the resizing stage before estimate the features. If\
    compression rate is 1 then the original image is used [Default = 0.2].

    :param str classifier: String representing the classifier to be used. By\
    now only SVM is implemented to keep the hand detector behaviour stable. 

    :ivar str LOG_TAG: Tag to be used for debuging purposes

    :ivar str feature: String representing the feature that is being used.

    :ivar FeatureController FeatureController: Instance to be used to extract\
    the features from the frames.

    :ivar Classifier classifier: sklearn classifier.

    :ivar Scaler scaler: sklearn scaler.

    * Example 1::

        import egovision
        from egovision import Video
        from egovision.handDetection import HandDetectionDataManager, DynamicHandDetector
        from egovision.extras import ObjectPickler
        filename = 'test_dm.pk'
        videoname = 'egovision/dataExamples/BENCHTEST.MP4'
        dm = ObjectPickler.load(HandDetectionDataManager, filename)
        hd = DynamicHandDetector("HOG",200,"SVM")
        hd.trainClassifier(dm)
        video = Video(videoname)
        hands = hd.classifyVideo(video,dtype="integer")
        ObjectPickler.save(hd, "test_hd.pk")
        print hands


    * Example 2::
    
        import egovision
        from egovision import Video
        from egovision.handDetection import DynamicHandDetector
        from egovision.extras import ObjectPickler
        videoname = 'egovision/dataExamples/BENCHTEST.MP4'
        video = Video(videoname)
        hd = ObjectPickler.load(DynamicHandDetector, detectorFilename)
        hd.setOptimalParameters(50.0)
        hands = hd.classifyVideo(video,dtype="integer")
        print hands

    """
    def __init__(self, feature, compressionWidth, classifier):
        if classifier == "SVM":
            HandDetector.__init__(self, feature, compressionWidth, classifier)
        else:
            from exceptions import UnavailableClassifier
            raise UnavailableClassifier("Sorry, {0} is not implemented yet!".format(classifier))
        self.TAG_LOG = "[Dynamic Hand Detector] "
        self.th = 0.5
        self.setOptimalParameters(50.0) #Default sampling rate

    def setOptimalParameters(self,dt):
        A, H, Q, R, X_0, T = self.getOptimalParameters(dt)
        self.setDynamicParameters(A, H, Q, R, X_0, T)
        

    def getOptimalParameters(self,dt):
        from egovision.handDetection.optimalParameters import getParameters
        A_dict, H_dict, Q_dict, R_dict, T_dict = getParameters(dt)
        if self.feature == "HOG":
            aux_feature = self.feature
        else:
            print self.TAG_LOG + "There is not estimate about the dynamic parameters for \
this feature. Currently there is only parameters for HOG-SVM. However, the \
hand detection is going to be evaluated with the default parameters"
            aux_feature = "DEFAULT"

        if isinstance(self.classifier,SVC):
            aux_classifier = "SVM"
        A = A_dict[aux_feature][aux_classifier]
        H = H_dict[aux_feature][aux_classifier]
        Q = Q_dict[aux_feature][aux_classifier]
        R = R_dict[aux_feature][aux_classifier]
        X_0 = 1
        T = T_dict[aux_feature][aux_classifier]
        return A, H, Q, R, X_0, T


    def setDynamicParameters(self,A,H,Q,R,X_0,th):
        """
        
        Define the parameters of the Dynamic Filter and the decision boundary.

        :ivar numpy.array(2x2) A: Process Matrix.

        :ivar numpy.array(2x1) H: Measurement Matrix.

        :ivar numpy.array(2x2) Q: Process Covariance Matrix.

        :ivar numpy.array(2x2) R: Measurement Covariance Matrix.

        :ivar float X_0: Initial Point.

        :ivar float th: Decision boundary.

        :returns: [HandDetectionDataManager] HandDetectionDataManager previously saved
        
        """
        
        self.kalman = df.KalmanFilter(A,H,Q,R,X_0)
        self.th = th

    def classifyVideo(self,video,dtype="integer"):
        """

        This method perform a full scan detection over a video. The result is a
        numpy.darray with the detection result. The result could be a binarized
        array (dtype="integer") or a real number(dtype="float") with
        positive(negative) value if the hands are present(absent) in each of
        the frames. 

        :ivar Video video: Video object

        :ivar str dtype: Type of result expected. "float" is only available if\
        the classifier is an SVM.

        * Example 1::
        

            import egovision
            from egovision import Video
            from egovision.handDetection import HandDetectionDataManager, DynamicHandDetector
            from egovision.extras import ObjectPickler
            filename = 'test_dm.pk'
            videoname = 'egovision/dataExamples/BENCHTEST.MP4'
            dm = ObjectPickler.load(HandDetectionDataManager, filename)
            hd = DynamicHandDetector("HOG",200,"SVM")
            hd.trainClassifier(dm)
            video = Video(videoname)
            hands = hd.classifyVideo(video,dtype="integer")
            ObjectPickler.save(hd, "test_hd.pk")
            print hands

        """

        return HandDetector.classifyVideo(self, video, dtype)

    def classifyFrame(self,frame,dtype="integer"):
        """

        This method detects the hands in a Frame and return: 1) a binarized
        array (dtype="integer") or, 2) a real number(dtype="float") with
        positive(negative) value if the hands are present(absent) in each of
        the frames. 

        :ivar Frame frame: Frame to be processed

        :ivar str dtype: Type of result expected. "float" is only available if\
        the classifier is an SVM.

        * Example 1::
        
            import egovision
            from egovision import Video
            from egovision.handDetection import HandDetectionDataManager, DynamicHandDetector
            from egovision.extras import ObjectPickler
            filename = 'test_dm.pk'
            videoname = 'egovision/dataExamples/BENCHTEST.MP4'
            dm = ObjectPickler.load(HandDetectionDataManager, filename)
            hd = DynamicHandDetector("HOG",200,"SVM")
            hd.trainClassifier(dm)
            video = Video(videoname)
            hands = hd.classifyFrame(video.next(),dtype="integer")
            ObjectPickler.save(hd, "test_hd.pk")
            print hands

        """
        return HandDetector.classifyFrame(self, frame, dtype)


    def classifyFeatureVideo(self, featureVideo,dtype="integer"):
        """

        This method detects the hands in a Feature Video and returns: 1) a
        binarized array (dtype="integer") or, 2) a real number(dtype="float")
        with positive(negative) value if the hands are present(absent) in each
        of the frames. 

        The main advantage of this function over classifyFrame is that if the
        experiment is well designed then this method does not requiere to
        estimate the features each time.

        :ivar FeatureVideo featureVideo: Feature object to be processed

        :ivar str dtype: Type of result expected. "float" is only available if\
        the classifier is an SVM.

        * Example 1::
        
            import egovision
            from egovision.handDetection import DynamicHandDetector
            from egovision.features import FeatureVideo
            from egovision.extras import ObjectPickler
            videoname = 'egovision/dataExamples/BENCHTEST.MP4'
            detectorFilename = 'egovision/dataExamples/UNIGEmin/handDetectors/GroundTruths/RGB_SVM.pk'
            fvname = 'egovision/dataExamples/GroundTruths/features/BENCHTEST_RGB_dynamic.pk'
            fv = ObjectPickler.load(FeatyreVideo, fvname)
            hd = ObjectPickler.load(DynamicHandDetector, detectorFilename)
            hd.setOptimalParameters(50.0)
            hands = hd.classifyFeatureVideo(fv,dtype="integer")
            print hands
            hands = hd.classifyFeatureVideo(fv,dtype="float")
            print hands

        """

        return HandDetector.classifyFeatureVideo(self, featureVideo, dtype)

    def classifyFeatureVector(self,feature,dtype="integer"):
        """

        This method detects the hands in a Feature vector of a frame and
        returns: 1) a binarized array (dtype="integer") or, 2) a real
        number(dtype="float") with positive(negative) value if the hands are
        present(absent) in each of the frames. 

        The main advantage of this function over classifyFrame is that if the
        experiment is well designed then this method does not requiere to
        estimate the features each time.

        :ivar Feature feature: Feature object to be processed

        :ivar str dtype: Type of result expected. "float" is only available if\
        the classifier is an SVM.

        * Example 1::
        
            import egovision
            from egovision import Video
            from egovision.handDetection import DynamicHandDetector
            from egovision.extras import ObjectPickler
            detectorFilename = 'egovision/dataExamples/UNIGEmin/handDetectors/GroundTruths/RGB_SVM_dynamic.pk'
            videoname = 'egovision/dataExamples/UNIGEmin/Videos/UNIGEmin_BENCHHANDS1.MP4'
            hd = ObjectPickler.load(DynamicHandDetector, detectorFilename)
            hd.setOptimalParameters(50.0)
            video = Video(videoname)
            frame = video.next()
            success, featureVideo = hd.featureController.getFeatures(frame)
            feature = featureVideo.next()
            hands = hd.classifyFeatureVector(feature,dtype="integer")
            print "integer", hands
            hands = hd.classifyFeatureVector(feature,dtype="float")
            print "float", hands
        """
        try:
            descriptor = self.scaler.transform(feature.astype(numpy.float).reshape(1,-1))
        except ValueError as e:
            from exceptions import SizeError
            raise SizeError(self.TAG_LOG + "The size of the feature vector does not \
match, Verify size of the features")
        measurement = self.__floatPredict__(descriptor)
        state = self.kalman.nextStep(measurement)

        if dtype == "integer":
            result = numpy.array([int(state[0:1] > self.th)])
        elif dtype == "float":
            result = state[0:1]
        return result

        
