from sklearn.ensemble import RandomForestRegressor
from gpuTree import GPUTree
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray


class GPURandomForestRegressorV2(RandomForestRegressor):
    
    def __init__(self, *args, **kargs):

        RandomForestRegressor.__init__(self, *args, **kargs)
        self.weight = np.array([1]).astype(np.float32)
        self.weight_gpu = cuda.mem_alloc(self.weight.nbytes)


    def __setKernel__(self):
        self.children_left = np.array(self.children_left).astype(np.uint32)
        self.children_right = np.array(self.children_right).astype(np.uint32)
        self.threshold = np.array(self.threshold).astype(np.int32)
        self.feature = np.array(self.feature).astype(np.int32)
        self.value = np.array(self.value).astype(np.float32)
        self.firstElements = np.array(self.firstElements).astype(np.int32)

        self.children_left_gpu = cuda.mem_alloc(self.children_left.nbytes)
        cuda.memcpy_htod(self.children_left_gpu, self.children_left)

        self.children_right_gpu = cuda.mem_alloc(self.children_right.nbytes)
        cuda.memcpy_htod(self.children_right_gpu, self.children_right)

        self.threshold_gpu = cuda.mem_alloc(self.threshold.nbytes)
        cuda.memcpy_htod(self.threshold_gpu, self.threshold)

        self.feature_gpu = cuda.mem_alloc(self.feature.nbytes)
        cuda.memcpy_htod(self.feature_gpu, self.feature)

        self.value_gpu = cuda.mem_alloc(self.value.nbytes)
        cuda.memcpy_htod(self.value_gpu, self.value)

        self.firstElements_gpu = cuda.mem_alloc(self.firstElements.nbytes)
        cuda.memcpy_htod(self.firstElements_gpu, self.firstElements)

        self.values_gpu = None
        self.initialize_values = True
        self.gridHeight = None
        self.gridWidth = None
        self.blockWidth = 32
        self.blockHeight = 32





        self.module = SourceModule("""
              __global__ void predict(int *featureList,
                                      float *weight,
                                      float *values,
                                      int *feature,
                                      int *threshold,
                                      float *value,
                                      uint *children_left,
                                      uint *children_right,
                                      int *firstElements,
                                      int height, int width)
              {
                  int idx = threadIdx.x + blockDim.x*blockIdx.x;
                  int idy = threadIdx.y + blockDim.y*blockIdx.y;
                  int featValue;
                  int node;
                  int firstIndex;
                  if (idx < width and idy < height) {
                     int id = width*idy + idx;
                     values[id] = 0.0;
                     for (int i=0; i < sizeof(firstElements) - 1; ++i) {
                          firstIndex = firstElements[i];
                          node = firstIndex;
                          while (children_left[node] != -1) {
                              featValue = featureList[id*3+ feature[node]];
                              if (featValue <= threshold[node]){
                                  node = children_left[node] + firstIndex; 
                              } else {
                                  node = children_right[node] + firstIndex; 
                              }
                          }
                          values[id] = values[id] + value[node]*weight[0];
                     }
                  }
             }
          """)
        
        self.func = self.module.get_function("predict")
    
    def predict(self, featureList_gpu, weight):
        weight = weight/float(len(self.firstElements)-1)
        self.weight = np.array([weight]).astype(np.float32)
        cuda.memcpy_htod(self.weight_gpu, self.weight)
        
        if self.initialize_values:
            self.values_gpu = gpuarray.zeros(featureList_gpu.size/3,np.float32)
            self.gridWidth = int(np.ceil(featureList_gpu.smp_width/float(self.blockWidth)))
            self.gridHeight = int(np.ceil(featureList_gpu.smp_height/float(self.blockHeight)))
            self.initialize_values = False
        
        self.func(featureList_gpu,
                  self.weight_gpu,
                  self.values_gpu,
                  self.feature_gpu,
                  self.threshold_gpu,
                  self.value_gpu,
                  self.children_left_gpu,
                  self.children_right_gpu,
                  self.firstElements_gpu,
                  np.int32(featureList_gpu.smp_height),
                  np.int32(featureList_gpu.smp_width),
                  block=(self.blockWidth,self.blockHeight,1),
                  grid=(self.gridWidth,self.gridHeight))
        
        return self.values_gpu

    def fit(self, *args, **kargs):
        RandomForestRegressor.fit(self, *args, **kargs)
        self.gpuEstimators_ = []
        self.children_left = []
        self.children_right = []
        self.threshold = []
        self.feature = []
        self.value = []
        self.firstElements = [0]
        for e in self.estimators_:
            self.children_left = np.hstack([self.children_left, e.tree_.children_left])
            self.children_right = np.hstack([self.children_right, e.tree_.children_right])
            self.threshold = np.hstack([self.threshold, e.tree_.threshold])
            self.feature = np.hstack([self.feature, e.tree_.feature])
            self.value = np.hstack([self.value, e.tree_.value.flatten()])
            self.firstElements.append(self.firstElements[-1] + e.tree_.children_left.shape[0])
        self.__setKernel__()



class GPURandomForestRegressor(RandomForestRegressor):
    
    def __init__(self, *args, **kargs):
        RandomForestRegressor.__init__(self, *args, **kargs)
        self.weight = np.array([1]).astype(np.float32)
        self.weight_gpu = cuda.mem_alloc(self.weight.nbytes)
    
    def predict(self, X, weight):
        weight = weight/float(len(self.estimators_))
        self.weight = np.array([weight]).astype(np.float32)
        cuda.memcpy_htod(self.weight_gpu, self.weight)
        
        y_tot = None
        for e in self.gpuEstimators_:
            result = e.predict(X, self.weight_gpu)
            if y_tot is None:
                y_tot = result
            else:
                y_tot += result
        return y_tot
        

    def fit(self, *args, **kargs):
        RandomForestRegressor.fit(self, *args, **kargs)
        self.gpuEstimators_ = []
        for e in self.estimators_:
            self.gpuEstimators_.append(GPUTree(e.tree_.children_left,
                                               e.tree_.children_right,
                                               e.tree_.threshold,
                                               e.tree_.feature,
                                               e.tree_.value))
