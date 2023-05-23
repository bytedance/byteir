#include <torch/script.h>

torch::Tensor custom_dynamic_stitch(std::vector<torch::Tensor> indices,
                                    std::vector<torch::Tensor> data) {
  int n = 0;
  for (auto &idx : indices) {
    n += idx.numel();
  }
  std::vector<torch::Tensor> res(n);
  for (size_t i = 0; i < data.size(); ++i) {
    for (int j = 0; j < indices[i].size(0); ++j) {
      res[indices[i][j].item<int>()] = data[i][j].unsqueeze(0);
    }
  }
  return torch::cat(res, /*dim=*/0);
}

static auto registry =
    torch::RegisterOperators("custom::dynamic_stitch", &custom_dynamic_stitch);
