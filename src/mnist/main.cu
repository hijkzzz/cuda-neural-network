#include <dataset.cuh>
#include <mnist.cuh>

#define BATCH_SIZE 64
#define LEARNING_RATE 0.01
#define L2 0
#define EPOCHS 30
#define BETA 0.99

int main() {
  //DataSet dataset("../mnist_data", true);
  //dataset.forward(64, true);
  //dataset.print_im();

  Minist mnist("../mnist_data", LEARNING_RATE, L2, BETA);
  mnist.train(EPOCHS, BATCH_SIZE);
}