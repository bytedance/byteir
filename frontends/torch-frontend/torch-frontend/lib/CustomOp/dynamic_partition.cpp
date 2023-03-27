#include <torch/script.h>

std::vector<torch::Tensor> custom_dynamic_partition(torch::Tensor data,
                                                    torch::Tensor partitions,
                                                    int64_t num_partitions) {
  std::vector<torch::Tensor> res;
  for (int i = 0; i < num_partitions; i++) {
    std::vector<int64_t> indices;
    for (int j = 0; j < partitions.size(0); j++) {
      if (partitions[j].item<int>() == i) {
        indices.push_back(j);
      }
    }
    res.push_back(data.index_select(
        0, torch::from_blob(indices.data(), {static_cast<long>(indices.size())},
                            torch::kLong)
               .clone()
               .to(data.device())));
  }
  return res;
}

static auto registry = torch::RegisterOperators("custom::dynamic_partition",
                                                &custom_dynamic_partition);
