#include <dataset.cuh>
#include <minist.cuh>

#define BATCH_SIZE 64
#define LEARNING_RATE 0.01
#define L2 0.001
#define EPOCHS 10
#define BETA 0.99

int main() {
  Minist minist("../mnist_data");
  minist.train(LEARNING_RATE, L2, BATCH_SIZE, EPOCHS, BETA);
}