# neural-network
Convolutional Neural Network with CUDA

## Layers
* Linear
* Conv2D
* MaxPool2D
* ReLU
* Softmax
* Sigmoid
* NLLLoss

## Optimizer
* RMSProp

## Prerequisites
* CMake 3.8+
* MSVC14.00/GCC6+
* CUDA9.0+

## Run
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build

# Download the minist dataset from http://yann.lecun.com/exdb/mnist/
./minist
```

## References
* [High Performance Convolutional Neural Networks for Document Processing](https://hal.inria.fr/file/index/docid/112631/filename/p1038112283956.pdf)
* [卷积神经网络(CNN)反向传播算法](https://www.cnblogs.com/pinard/p/6494810.html)
* [矩阵求导术](https://zhuanlan.zhihu.com/p/24709748)
* Caffe
* CUDA Toolkit Documents
