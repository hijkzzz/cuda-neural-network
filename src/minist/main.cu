#include <dataset.cuh>
#include <minist.cuh>

void display_image(const std::vector<float> &image, int label) {
  std::cout << label << std::endl;
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      std::cout << (image[i * 28 + j] > 0.5 ? "*" : " ");
    }
    std::cout << std::endl;
  }
}

int main() {
  // test dataset
  DataSet dataset("../minist_data");
  auto train_data = dataset.get_train_data(32);

  // show images
  for (int i = 0; i < 3; i++) {
    display_image(train_data.first[i], train_data.second[i]);
  }
}