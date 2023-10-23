/**
 * app.c
 * math Host Application Source File
 *
 */

#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "brt/backends/pim/upmem/device/common.h"
#include "./elementwise_host.h"
#include "brt/core/ir/ir.h"
#include "brt/core/ir/util.h"
// Define the DPU Binary path as DPU_BINARY here

namespace brt {
namespace pim {

namespace upmem {
namespace kernel {
int runadd(dpu_set_t *dpu_set, dpu_set_t dpu, uint32_t nr_of_dpus, void *A,
           void *B, void *C, int m, int n) {

  unsigned int i;

  // Initialize help data
  dpu_info =
      (struct dpu_info_t *)malloc(nr_of_dpus * sizeof(struct dpu_info_t));
  dpu_arguments_t *input_args =
      (dpu_arguments_t *)malloc(nr_of_dpus * sizeof(dpu_arguments_t));
  uint32_t max_rows_per_dpu = 0;
  uint32_t n_size_pad = n_size;
  if (n_size % 2 == 1) {
    n_size_pad++;
  }

  i = 0;
  DPU_FOREACH(dpu_set, dpu, i) {
    uint32_t rows_per_dpu;
    uint32_t prev_rows_dpu = 0;
    uint32_t chunks = m_size / nr_of_dpus;
    rows_per_dpu = chunks;
    uint32_t rest_rows = m_size % nr_of_dpus;
    if (i < rest_rows)
      rows_per_dpu++;
    if (rest_rows > 0) {
      if (i >= rest_rows)
        prev_rows_dpu = rest_rows * (chunks + 1) + (i - rest_rows) * chunks;
      else
        prev_rows_dpu = i * (chunks + 1);
    } else {
      prev_rows_dpu = i * chunks;
    }

    // Keep max rows for parallel transfers
    uint32_t rows_per_dpu_pad = rows_per_dpu;
    if (rows_per_dpu_pad % 2 == 1) // 4-byte elements
      rows_per_dpu_pad++;
    if (rows_per_dpu_pad > max_rows_per_dpu)
      max_rows_per_dpu = rows_per_dpu_pad;

    dpu_info[i].rows_per_dpu = rows_per_dpu;
    dpu_info[i].rows_per_dpu_pad = rows_per_dpu_pad;
    dpu_info[i].prev_rows_dpu = prev_rows_dpu;

    // Copy input arguments to DPU
    input_args[i].n_size = n_size;
    input_args[i].n_size_pad = n_size_pad;
    input_args[i].nr_rows = rows_per_dpu;
  }

  for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

    // Input arguments
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      // Copy input arguments to DPU
      input_args[i].max_rows = max_rows_per_dpu;

      DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,
                             sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

    // Copy input array and vector
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, A + dpu_info[i].prev_rows_dpu * n_size));
    }
    DPU_ASSERT(dpu_push_xfer(
        dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
        max_rows_per_dpu * n_size_pad * sizeof(T), DPU_XFER_DEFAULT));
    DPU_FOREACH(dpu_set, dpu, i) { DPU_ASSERT(dpu_prepare_xfer(dpu, B)); }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                             DPU_MRAM_HEAP_POINTER_NAME,
                             max_rows_per_dpu * n_size_pad * sizeof(T),
                             n_size_pad * sizeof(T), DPU_XFER_DEFAULT));

    // Run kernel on DPUs

    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, C + i * max_rows_per_dpu));
    }
    DPU_ASSERT(dpu_push_xfer(
        dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
        max_rows_per_dpu * n_size_pad * sizeof(T) + n_size_pad * sizeof(T),
        max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
  }

  // // Deallocation
  // free(A);
  // free(B);
  // free(C);
  // free(C_dpu);
  DPU_ASSERT(dpu_free(dpu_set));

  return status 0;
}

int runsub(dpu_set_t *dpu_set, dpu_set_t dpu, uint32_t nr_of_dpus, void *A,
           void *B, void *C, int m, int n) {

  unsigned int i;

  // Initialize help data
  dpu_info =
      (struct dpu_info_t *)malloc(nr_of_dpus * sizeof(struct dpu_info_t));
  dpu_arguments_t *input_args =
      (dpu_arguments_t *)malloc(nr_of_dpus * sizeof(dpu_arguments_t));
  uint32_t max_rows_per_dpu = 0;
  uint32_t n_size_pad = n_size;
  if (n_size % 2 == 1) {
    n_size_pad++;
  }

  i = 0;
  DPU_FOREACH(dpu_set, dpu, i) {
    uint32_t rows_per_dpu;
    uint32_t prev_rows_dpu = 0;
    uint32_t chunks = m_size / nr_of_dpus;
    rows_per_dpu = chunks;
    uint32_t rest_rows = m_size % nr_of_dpus;
    if (i < rest_rows)
      rows_per_dpu++;
    if (rest_rows > 0) {
      if (i >= rest_rows)
        prev_rows_dpu = rest_rows * (chunks + 1) + (i - rest_rows) * chunks;
      else
        prev_rows_dpu = i * (chunks + 1);
    } else {
      prev_rows_dpu = i * chunks;
    }

    // Keep max rows for parallel transfers
    uint32_t rows_per_dpu_pad = rows_per_dpu;
    if (rows_per_dpu_pad % 2 == 1) // 4-byte elements
      rows_per_dpu_pad++;
    if (rows_per_dpu_pad > max_rows_per_dpu)
      max_rows_per_dpu = rows_per_dpu_pad;

    dpu_info[i].rows_per_dpu = rows_per_dpu;
    dpu_info[i].rows_per_dpu_pad = rows_per_dpu_pad;
    dpu_info[i].prev_rows_dpu = prev_rows_dpu;

    // Copy input arguments to DPU
    input_args[i].n_size = n_size;
    input_args[i].n_size_pad = n_size_pad;
    input_args[i].nr_rows = rows_per_dpu;
  }

  for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

    // Input arguments
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      // Copy input arguments to DPU
      input_args[i].max_rows = max_rows_per_dpu;

      DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,
                             sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

    // Copy input array and vector
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, A + dpu_info[i].prev_rows_dpu * n_size));
    }
    DPU_ASSERT(dpu_push_xfer(
        dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
        max_rows_per_dpu * n_size_pad * sizeof(T), DPU_XFER_DEFAULT));
    DPU_FOREACH(dpu_set, dpu, i) { DPU_ASSERT(dpu_prepare_xfer(dpu, B)); }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                             DPU_MRAM_HEAP_POINTER_NAME,
                             max_rows_per_dpu * n_size_pad * sizeof(T),
                             n_size_pad * sizeof(T), DPU_XFER_DEFAULT));

    // Run kernel on DPUs

    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, C + i * max_rows_per_dpu));
    }
    DPU_ASSERT(dpu_push_xfer(
        dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
        max_rows_per_dpu * n_size_pad * sizeof(T) + n_size_pad * sizeof(T),
        max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
  }

  DPU_ASSERT(dpu_free(dpu_set));

  return status 0;
}
int rundiv(dpu_set_t *dpu_set, dpu_set_t dpu, uint32_t nr_of_dpus, void *A,
           void *B, void *C, int m, int n) {

  unsigned int i;

  // Initialize help data
  dpu_info =
      (struct dpu_info_t *)malloc(nr_of_dpus * sizeof(struct dpu_info_t));
  dpu_arguments_t *input_args =
      (dpu_arguments_t *)malloc(nr_of_dpus * sizeof(dpu_arguments_t));
  uint32_t max_rows_per_dpu = 0;
  uint32_t n_size_pad = n_size;
  if (n_size % 2 == 1) {
    n_size_pad++;
  }

  i = 0;
  DPU_FOREACH(dpu_set, dpu, i) {
    uint32_t rows_per_dpu;
    uint32_t prev_rows_dpu = 0;
    uint32_t chunks = m_size / nr_of_dpus;
    rows_per_dpu = chunks;
    uint32_t rest_rows = m_size % nr_of_dpus;
    if (i < rest_rows)
      rows_per_dpu++;
    if (rest_rows > 0) {
      if (i >= rest_rows)
        prev_rows_dpu = rest_rows * (chunks + 1) + (i - rest_rows) * chunks;
      else
        prev_rows_dpu = i * (chunks + 1);
    } else {
      prev_rows_dpu = i * chunks;
    }

    // Keep max rows for parallel transfers
    uint32_t rows_per_dpu_pad = rows_per_dpu;
    if (rows_per_dpu_pad % 2 == 1) // 4-byte elements
      rows_per_dpu_pad++;
    if (rows_per_dpu_pad > max_rows_per_dpu)
      max_rows_per_dpu = rows_per_dpu_pad;

    dpu_info[i].rows_per_dpu = rows_per_dpu;
    dpu_info[i].rows_per_dpu_pad = rows_per_dpu_pad;
    dpu_info[i].prev_rows_dpu = prev_rows_dpu;

    // Copy input arguments to DPU
    input_args[i].n_size = n_size;
    input_args[i].n_size_pad = n_size_pad;
    input_args[i].nr_rows = rows_per_dpu;
  }

  for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

    // Input arguments
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      // Copy input arguments to DPU
      input_args[i].max_rows = max_rows_per_dpu;

      DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,
                             sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

    // Copy input array and vector
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, A + dpu_info[i].prev_rows_dpu * n_size));
    }
    DPU_ASSERT(dpu_push_xfer(
        dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
        max_rows_per_dpu * n_size_pad * sizeof(T), DPU_XFER_DEFAULT));
    DPU_FOREACH(dpu_set, dpu, i) { DPU_ASSERT(dpu_prepare_xfer(dpu, B)); }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                             DPU_MRAM_HEAP_POINTER_NAME,
                             max_rows_per_dpu * n_size_pad * sizeof(T),
                             n_size_pad * sizeof(T), DPU_XFER_DEFAULT));

    // Run kernel on DPUs

    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, C + i * max_rows_per_dpu));
    }
    DPU_ASSERT(dpu_push_xfer(
        dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
        max_rows_per_dpu * n_size_pad * sizeof(T) + n_size_pad * sizeof(T),
        max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
  }

  // // Deallocation
  // free(A);
  // free(B);
  // free(C);
  // free(C_dpu);
  DPU_ASSERT(dpu_free(dpu_set));

  return status 0;
}
int runmultiply(dpu_set_t *dpu_set, dpu_set_t dpu, uint32_t nr_of_dpus, T *A,
                T *B, T *C, int m, int n) {

  unsigned int i;

  // Initialize help data
  dpu_info =
      (struct dpu_info_t *)malloc(nr_of_dpus * sizeof(struct dpu_info_t));
  dpu_arguments_t *input_args =
      (dpu_arguments_t *)malloc(nr_of_dpus * sizeof(dpu_arguments_t));
  uint32_t max_rows_per_dpu = 0;
  uint32_t n_size_pad = n_size;
  if (n_size % 2 == 1) {
    n_size_pad++;
  }

  i = 0;
  DPU_FOREACH(dpu_set, dpu, i) {
    uint32_t rows_per_dpu;
    uint32_t prev_rows_dpu = 0;
    uint32_t chunks = m_size / nr_of_dpus;
    rows_per_dpu = chunks;
    uint32_t rest_rows = m_size % nr_of_dpus;
    if (i < rest_rows)
      rows_per_dpu++;
    if (rest_rows > 0) {
      if (i >= rest_rows)
        prev_rows_dpu = rest_rows * (chunks + 1) + (i - rest_rows) * chunks;
      else
        prev_rows_dpu = i * (chunks + 1);
    } else {
      prev_rows_dpu = i * chunks;
    }

    // Keep max rows for parallel transfers
    uint32_t rows_per_dpu_pad = rows_per_dpu;
    if (rows_per_dpu_pad % 2 == 1) // 4-byte elements
      rows_per_dpu_pad++;
    if (rows_per_dpu_pad > max_rows_per_dpu)
      max_rows_per_dpu = rows_per_dpu_pad;

    dpu_info[i].rows_per_dpu = rows_per_dpu;
    dpu_info[i].rows_per_dpu_pad = rows_per_dpu_pad;
    dpu_info[i].prev_rows_dpu = prev_rows_dpu;

    // Copy input arguments to DPU
    input_args[i].n_size = n_size;
    input_args[i].n_size_pad = n_size_pad;
    input_args[i].nr_rows = rows_per_dpu;
  }

  for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

    // Input arguments
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      // Copy input arguments to DPU
      input_args[i].max_rows = max_rows_per_dpu;

      DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,
                             sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

    // Copy input array and vector
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, A + dpu_info[i].prev_rows_dpu * n_size));
    }
    DPU_ASSERT(dpu_push_xfer(
        dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0,
        max_rows_per_dpu * n_size_pad * sizeof(T), DPU_XFER_DEFAULT));
    DPU_FOREACH(dpu_set, dpu, i) { DPU_ASSERT(dpu_prepare_xfer(dpu, B)); }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                             DPU_MRAM_HEAP_POINTER_NAME,
                             max_rows_per_dpu * n_size_pad * sizeof(T),
                             n_size_pad * sizeof(T), DPU_XFER_DEFAULT));

    // Run kernel on DPUs

    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, C + i * max_rows_per_dpu));
    }
    DPU_ASSERT(dpu_push_xfer(
        dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME,
        max_rows_per_dpu * n_size_pad * sizeof(T) + n_size_pad * sizeof(T),
        max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
  }

  // // Deallocation
  // free(A);
  // free(B);
  // free(C);
  // free(C_dpu);
  DPU_ASSERT(dpu_free(dpu_set));

  return status 0;
}

} // namespace kernel
} // namespace upmem
} // namespace pim
} // namespace brt