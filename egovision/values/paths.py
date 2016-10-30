VIDEO_EXAMPLE_PATH = "egovision/dataExamples/{0}"
GROUNDTRUTH_PATH = "egovision/dataExamples/GroundTruths/{0}"
GROUNDTRUTH_FEATURE_PATH = GROUNDTRUTH_PATH.format("features/{0}.pk")
GROUNDTRUTH_VIDEOFEATURE_PATH = GROUNDTRUTH_PATH.format("features/{0}_{1}.pk")
DATASET_PATH = "egovision/dataExamples/{0}"
DATASET_MASKS_PATH = DATASET_PATH + "/masks/{1}/"
DATASET_LABELS_PATH = DATASET_PATH + "/labels/{1}.txt"
DATASET_LRMASKS_PATH = DATASET_PATH + "/LRmasks/{1}/"
DATASET_FRAMES_PATH = DATASET_PATH + "/img/{1}/"

DATASET_POSITIVES_PATH = DATASET_PATH + "/Positives/"
DATASET_NEGATIVES_PATH = DATASET_PATH + "/Negatives/"
DATASET_GROUNDTRUTH_PATH = DATASET_PATH + "/GroundTruth/"
DATASET_VIDEOS_PATH = DATASET_PATH + "/Videos/{1}"
DATASET_DATAMANAGER_PATH = DATASET_PATH + "/dataManager/{1}.pk"
DATASET_DATAMANAGER_GT_PATH = DATASET_GROUNDTRUTH_PATH + "dataManager/{1}.pk"
DATASET_HANDDETECTOR_GT_PATH = DATASET_GROUNDTRUTH_PATH + "handDetector/{1}_{2}{3}.pk"
DATASET_HSDATAMANAGER_GT_PATH = DATASET_GROUNDTRUTH_PATH + "HandSegmenter/{1}_{2}_{3}.pk"
DATASET_MULTIHSDATAMANAGER_GT_PATH = DATASET_GROUNDTRUTH_PATH + "multiHandSegmenter/{1}_{2}_{3}.pk"
