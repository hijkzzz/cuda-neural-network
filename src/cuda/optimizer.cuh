#pragma once

#include <utility>
#include <vector>

#include <storage.cuh>

class Optimizer {
 public:
  virtual void step() = 0;
  virtual void regist(std::vector<std::pair<Storage *, Storage *>> params) = 0;

 protected:
  std::vector<Storage *> parameter_list;
  std::vector<Storage *> grad_list;
};