#include <torch/script.h>

torch::Tensor custom_dynamic_mask_stitch(std::vector<torch::Tensor> data,
                                         torch::Tensor partitions,
                                         std::vector<int64_t> output_shape) {
  std::vector<torch::Tensor> res;
  res.reserve(partitions.size(0));
  std::vector<size_t> count(data.size(), 0);
  for (int64_t i = 0; i < partitions.size(0); ++i) {
    int idx = partitions[i].item<int>();
    res.push_back(data[idx][count[idx]].unsqueeze(0));
    count[idx]++;
  }
  return torch::cat(res, /*dim=*/0);
}

static auto registry = torch::RegisterOperators("custom::dynamic_mask_stitch",
                                                &custom_dynamic_mask_stitch);
