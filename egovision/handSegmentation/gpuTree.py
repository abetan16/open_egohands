import numpy as np
import math
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

class GPUTree:
    def __init__(self, children_left, children_right, threshold, feature, value):
        self.children_left = np.array(children_left).astype(np.uint32)
        self.children_right = np.array(children_right).astype(np.uint32)
        self.threshold = np.array(threshold).astype(np.int32)
        self.feature = np.array(feature).astype(np.int32)
        self.value = np.array(value).astype(np.float32)

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
                                      int size)
              {
                  int idx = threadIdx.x + blockDim.x*blockIdx.x;
                  int idy = threadIdx.y + blockDim.y*blockIdx.y;
                  int featValue;
                  int id = blockDim.x*gridDim.x*idy + idx;
                  if (id < size) {
                      int node = 0;
                      while (children_left[node] != -1){
                          featValue = featureList[id*3+ feature[node]];
                          if (featValue <= threshold[node]){
                              node = children_left[node]; 
                          } else {
                              node = children_right[node]; 
                          }
                      }
                      values[id] = value[node]*weight[0];
                 }
             }
          """)
        
        self.func = self.module.get_function("predict")
        

    
    def predict(self, featureList_gpu, weight_gpu):
        


        if self.initialize_values:
            # self.values = np.zeros(len(featureList)).astype(np.float32)
            # self.values_gpu = cuda.mem_alloc(self.values.nbytes)

            # cuda.memcpy_htod(self.values_gpu, self.values)

            self.values_gpu = gpuarray.zeros(featureList_gpu.size/3,np.float32)
            self.gridWidth = int(np.ceil(featureList_gpu.smp_width/float(self.blockWidth)))
            self.gridHeight = int(np.ceil(featureList_gpu.smp_height/float(self.blockHeight)))
            self.initialize_values = False
            #self.featureList_gpu = cuda.mem_alloc(featureList.nbytes)

	#cuda.memcpy_htod(self.featureList_gpu, featureList)


        self.func(featureList_gpu,
                  weight_gpu,
                  self.values_gpu,
                  self.feature_gpu,
                  self.threshold_gpu,
                  self.value_gpu,
                  self.children_left_gpu,
                  self.children_right_gpu,
                  np.int32(self.values_gpu.size),
                  block=(self.blockHeight,self.blockWidth,1),
                  grid=(self.gridHeight,self.gridWidth))
                  
        # values_result = np.empty_like(self.values)
        # cuda.memcpy_dtoh(values_result, self.values_gpu)
        return self.values_gpu

