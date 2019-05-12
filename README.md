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

# Download the mnist dataset from http://yann.lecun.com/exdb/mnist/
# Unzip and put to ./mnist_data
./mnist
```

## Performance
```
conv 1 32 5 relu
maxpool 2
conv 32 64 5 relu
maxpool 2
conv 64 128 3 relu
fc 4 * 128 128 relu
fc 128 10 relu
softmax

shuffle = true
batch_size = 128
learning_rate = 0.003
L2 = 0.0001
beta = 0.99
```

* 1 epoch 93%
* 10 epochs 99.12%
* 30 epochs 99.23%
* 10s / epoch(GTX1070)

## TODO
* Faster matmul kernel function
* CUDA Streams

## References
* [High Performance Convolutional Neural Networks for Document Processing](https://hal.inria.fr/file/index/docid/112631/filename/p1038112283956.pdf)
* [卷积神经网络(CNN)反向传播算法](https://www.cnblogs.com/pinard/p/6494810.html)
* [矩阵求导术](https://zhuanlan.zhihu.com/p/24709748)
* Caffe
* CUDA Toolkit Documents
