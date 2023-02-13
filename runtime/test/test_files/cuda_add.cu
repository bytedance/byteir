extern "C" __global__ void nvrtc_add_kernel(const float* input, float* output, int n, float val) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = input[i]+ val;
  }
}