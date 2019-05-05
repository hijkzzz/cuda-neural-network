#include <dataset.cuh>
#include <minist.cuh>

int main() {
  // test dataset
  DataSet dataset("../minist_data");
  auto train_data = dataset.get_train_data(32);
  auto test_data = dataset.get_test_data(32);
}