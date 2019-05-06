#include <dataset.cuh>
#include <minist.cuh>

#define BATCH_SIZE 64
#define LEARNING_RATE 0.01
#define L2 0.001
#define EPOCHS 30
#define BETA 0.99

void show_image(const std::vector<float> &image, int label, int height,
                int width) {
  std::cout << label << std::endl;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      std::cout << (image[i * width + j] > 0 ? "* " : "  ");
    }
    std::cout << std::endl;
  }
}

int main() {
  DataSet dataset("../mnist_data", true);
  Minist minist(&dataset);

  // show images
  // auto train_data = dataset.get_train_data(BATCH_SIZE);
  //  for (int i = 0; i < train_data.size(); i++) {
  //    show_image(train_data.first[i], train_data.second[i],
  //    dataset.get_height(), dataset.get_width());
  //  }

  minist.train(LEARNING_RATE, L2, BATCH_SIZE, EPOCHS, BETA);
}