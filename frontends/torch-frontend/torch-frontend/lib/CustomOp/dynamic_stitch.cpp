#include <torch/script.h>

torch::Tensor custom_dynamic_stitch(std::vector<torch::Tensor> indices,
                                    std::vector<torch::Tensor> data,
                                    std::vector<int64_t> output_shape) {
  int n = 0;
  for (auto &idx : indices) {
    n += idx.numel();
  }
  std::vector<torch::Tensor> res(n);

  for (int i = 0; i < data.size(); ++i) {
    if (indices[i].size(0) == 0) {
      continue;
    }
    auto idx = indices[i].view({-1});
    auto d = data[i].view({idx.numel(), -1});
    for (int j = 0; j < idx.numel(); ++j) {
      res[idx[j].item<int>()] = d[j].unsqueeze(0);
    }
  }

  return torch::cat(res, /*dim=*/0);
}

static auto registry =
    torch::RegisterOperators("custom::dynamic_stitch", &custom_dynamic_stitch);
