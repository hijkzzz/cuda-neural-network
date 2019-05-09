#include <dataset.cuh>
#include <mnist.cuh>

#define BATCH_SIZE 64
#define LEARNING_RATE 0.01
#define L2 0.001
#define EPOCHS 10
#define BETA 0.99

int main() {
  Minist mnist("../mnist_data");
  mnist.train(EPOCHS, BATCH_SIZE, LEARNING_RATE, L2, BETA);
}