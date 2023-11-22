
#include "brt/backends/pim/samsung/device/dpu_call.h"
#include "brt/backends/pim/samsung/device/hbm_allocator.h"
#include "brt/core/framework/allocator.h"
#include "brt/test/common/util.h"
#include "gtest/gtest.h"
#include <stdlib.h>
#include <string>
using namespace brt;
using namespace brt::pim::hbmpim;

// BRT_HBMPIM_CALL_THRW

// void CheckCUDABuffer(T *mat, size_t size, std::function<void(T *)> check) {
//   T *h_ptr = (T *)malloc(size * sizeof(T));
// //   cudaMemcpy(h_ptr, mat, size * sizeof(T), cudaMemcpyDeviceToHost);
// //   cudaDeviceSynchronize();
//   cudaDeviceSynchronize();
//   check(h_ptr);
//   free(h_ptr);
// }

static void CheckResult(void *d_ptr, size_t size, char val) {
  brt::test::CheckValues<char>((char *)d_ptr, size, val);
}

static inline void test_func(IAllocator *hbmpimallocator) {
#if BRT_TEST_WITH_ASAN
  size_t large_size = 32 * 1024 * 1024;
#else
  size_t large_size = 1024 * 1024 * 1024;
#endif

  auto ptr = hbmpimallocator->Alloc(large_size);
  EXPECT_TRUE(ptr != nullptr);
  // test the bytes are ok for read/write
  memset(ptr, -1, large_size);
  CheckResult(ptr, large_size, -1);

  hbmpimallocator->Free(ptr);

  size_t small_size = 1024 * 1024; // 1MB
  size_t cnt = 1024;
  // check time for many allocation
  for (size_t s = 0; s < cnt; ++s) {
    void *raw = hbmpimallocator->Alloc(small_size);
    hbmpimallocator->Free(raw);
  }
  EXPECT_TRUE(true);
}

TEST(HBMAllocatorTest, HBMPIMBase) {

  HBMPIMAllocator hbmpim_base_alloc(0, "hbmpim"); // default HBM
  test_func(&hbmpim_base_alloc);
}
