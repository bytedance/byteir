/* Copyright 2020 UPMEM. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef DPU_H
#define DPU_H

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "dpu_checkpoint.h"
#include "dpu_error.h"
#include "dpu_macro_utils.h"
#include "dpu_types.h"

/**
 * @mainpage
 *
 * @section Overview
 *
 * This API provides features to manage the DPUs available on the host machine.
 *
 * To allocate DPUs, see the functions dpu_alloc() and dpu_alloc_ranks().
 *
 * Asynchronous operations on DPUs are available on some functions by setting
 * some specific flags as arguments.
 *
 * A program can be loaded and executed on the DPUs with the functions
 * dpu_load() and dpu_launch().
 *
 * Memory operations between the Host and the DPUs are done with the functions
 * dpu_broadcast_to(), dpu_prepare_xfer() and dpu_push_xfer().
 */

/**
 * @file dpu.h
 * @brief C API to manage DPUs.
 *
 * This API allows to get, pilot and share information with multiple DPUs.
 */

/**
 * @brief The different synchronization methods for launching DPUs.
 * polling method to check whether the execution is complete or not.
 */
typedef enum _dpu_launch_policy_t {
  /**
   * Do not wait for the DPUs to complete their execution.
   * The application is given back the control once the DPUs are booted and
   * shall use a synchronization
   */
  DPU_ASYNCHRONOUS,
  /** Suspend the application execution until the booted DPUs complete their
     work. */
  DPU_SYNCHRONOUS,
} dpu_launch_policy_t;

/**
 * @brief Direction for a DPU memory transfer.
 */
typedef enum _dpu_xfer_t {
  /** Transfer from Host to DPU. */
  DPU_XFER_TO_DPU,
  /** Transfer from DPU to Host. */
  DPU_XFER_FROM_DPU,
} dpu_xfer_t;

/**
 * @brief Options for a DPU memory transfer.
 */
typedef enum _dpu_xfer_flags_t {
  /** Memory transfer is executed and transfer buffer pointers are cleared. */
  DPU_XFER_DEFAULT = 0,
  /** Memory transfer is executed and transfer buffer pointers are not cleared
   */
  DPU_XFER_NO_RESET = 1 << 0,
  /**
   * Memory transfer is done asynchronously. The application is given back the
   * control once the transfer is enqueue in the asynchronous job list of the
   * rank(s).
   */
  DPU_XFER_ASYNC = 1 << 1,
} dpu_xfer_flags_t;

/**
 * @brief Options for a DPU scatter gather memory transfer.
 */
typedef enum _dpu_sg_xfer_flags_t {
  /** Memory transfer is executed and transfer buffer pointers are cleared. */
  DPU_SG_XFER_DEFAULT = 0,
  /**
   * Memory transfer is done asynchronously. The application is given back the
   * control once the transfer is enqueue in the asynchronous job list of the
   * rank(s).
   */
  DPU_SG_XFER_ASYNC = 1 << 1,
  /**
   * Authorize scatter/gather memory transfers with a total number of bytes less
   * than the value specified with the argument "length" (dpu_push_sg_xfer(..,
   * length,..)). If this happen during a DPU_XFER_TO_DPU transfer, the
   * remaining MRAM bytes will be filled with zeros.
   */
  DPU_SG_XFER_DISABLE_LENGTH_CHECK = 1 << 2,

} dpu_sg_xfer_flags_t;

/**
 * @brief Options for a DPU callback.
 */
typedef enum _dpu_callback_flags_t {
  /** Callback is synchronous and called for each rank of the DPU set. */
  DPU_CALLBACK_DEFAULT = 0,
  /** Callback is asynchronous and blocking. */
  DPU_CALLBACK_ASYNC = 1 << 0,
  /**
   * Callback does not block the API calls to the ranks of the DPU set.
   * Callback must be asynchronous for this to have an effect.
   */
  DPU_CALLBACK_NONBLOCKING = 1 << 1,
  /**
   * Callback is called once only with the whole DPU set as argument.
   * Callback must be asynchronous and non-blocking for this to have an effect.
   */
  DPU_CALLBACK_SINGLE_CALL = 1 << 2,
} dpu_callback_flags_t;

/**
 * @brief Error management for DPU api functions
 * @param statement the call to the DPU api to execute and check
 * @param on_error the statement to execute in case of an error in the DPU api
 * call
 * @hideinitializer
 */
#define DPU_CHECK(statement, on_error)                                         \
  do {                                                                         \
    dpu_error_t __error = (statement);                                         \
    if (__error != DPU_OK) {                                                   \
      fprintf(stderr, "%s:%d(%s): DPU Error (%s)\n", __FILE__, __LINE__,       \
              __func__, dpu_error_to_string(__error));                         \
      on_error;                                                                \
    }                                                                          \
  } while (0)

/**
 * @brief Error management for DPU api functions, calling exit if an error
 * happens
 * @param statement the call to the DPU api to execute and check
 * @hideinitializer
 */
#define DPU_ASSERT(statement) DPU_CHECK(statement, exit(EXIT_FAILURE))

/**
 * @brief Allow user to allocate all dpus available with function `dpu_alloc`
 * @hideinitializer
 */
#define DPU_ALLOCATE_ALL (UINT_MAX)

/**
 * @brief Allocate a number of ranks that will have the specified number of
 * DPUs.
 *
 * Fails if the given number of DPUs cannot be allocated (unless
 * `DPU_ALLOCATE_ALL` is used).
 *
 * @param nr_dpus number of DPUs to allocate. Use `DPU_ALLOCATE_ALL` to allocate
 * all available DPUs.
 * @param profile list of (key=value) separated by comma to specify what kind of
 * dpu to allocate. Use `NULL` for the default profile.
 * @param dpu_set storage for the DPU set
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_alloc(uint32_t nr_dpus, const char *profile,
                      struct dpu_set_t *dpu_set);

/**
 * @brief Allocate the specified number of DPU ranks.
 *
 * Fails if the given number of DPU ranks cannot be allocated (unless
 * `DPU_ALLOCATE_ALL` is used).
 *
 * @param nr_ranks number of DPU ranks to allocate. Use `DPU_ALLOCATE_ALL` to
 * allocate all available DPU ranks.
 * @param profile list of (key=value) separated by comma to specify what kind of
 * dpu to allocate. Use `NULL` for the default profile.
 * @param dpu_set storage for the DPU set
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_alloc_ranks(uint32_t nr_ranks, const char *profile,
                            struct dpu_set_t *dpu_set);

/**
 * @brief Free all the DPUs of a DPU set.
 *
 * Note that this function will fail if called with a DPU set not provided by
 * `dpu_alloc`.
 *
 * @param dpu_set the identifier of the freed DPU set
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_free(struct dpu_set_t dpu_set);

/**
 * @brief Fetch the number of DPU ranks in a given DPU set.
 * @param dpu_set the DPU set identifier
 * @param nr_ranks the number of DPU ranks in the DPU set
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_get_nr_ranks(struct dpu_set_t dpu_set, uint32_t *nr_ranks);

/**
 * @brief Fetch the number of DPUs in a given DPU set.
 * @param dpu_set the DPU set identifier
 * @param nr_dpus the number of DPUs in the DPU set
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_get_nr_dpus(struct dpu_set_t dpu_set, uint32_t *nr_dpus);

/**
 * @brief Iterator over all DPU ranks of a DPU set.
 * @param set the targeted DPU set
 * @param rank a pointer to a `struct dpu_set_t`, which will store the DPU rank
 * context for the current iteration
 * @hideinitializer
 */
#define DPU_RANK_FOREACH(set, rank, ...)                                       \
  _CONCAT(_DPU_RANK_FOREACH_, _DPU_FOREACH_VARIANT(set, rank, ##__VA_ARGS__))  \
  (set, rank, ##__VA_ARGS__)

/**
 * @brief Intenal macro for DPU_RANK_FOREACH without rank index.
 * @private
 * @hideinitializer
 */
#define _DPU_RANK_FOREACH_X(set, rank)                                         \
  for (struct dpu_set_rank_iterator_t __rank_it =                              \
           dpu_set_rank_iterator_from(&set);                                   \
       rank = __rank_it.next, __rank_it.has_next;                              \
       dpu_set_rank_iterator_next(&__rank_it))

/**
 * @brief Intenal macro for DPU_RANK_FOREACH with rank index.
 * @private
 * @hideinitializer
 */
#define _DPU_RANK_FOREACH_I(set, rank, i)                                      \
  for (struct dpu_set_rank_iterator_t __rank_it =                              \
           dpu_set_rank_iterator_from(&set);                                   \
       i = __rank_it.count, rank = __rank_it.next, __rank_it.has_next;         \
       dpu_set_rank_iterator_next(&__rank_it))

/**
 * @brief Iterator over all DPUs of a DPU set.
 * @param set the targeted DPU set
 * @param dpu a pointer to a `struct dpu_set_t`, which will store the dpu
 * context for the current iteration
 * @hideinitializer
 */
#define DPU_FOREACH(set, dpu, ...)                                             \
  _CONCAT(_DPU_FOREACH_, _DPU_FOREACH_VARIANT(set, dpu, ##__VA_ARGS__))        \
  (set, dpu, ##__VA_ARGS__)

/**
 * @brief Intenal macro for DPU_RANK_FOREACH without rank index.
 * @private
 * @hideinitializer
 */
#define _DPU_FOREACH_X(set, dpu)                                               \
  for (struct dpu_set_dpu_iterator_t __dpu_it =                                \
           dpu_set_dpu_iterator_from(&set);                                    \
       dpu = __dpu_it.next, __dpu_it.has_next;                                 \
       dpu_set_dpu_iterator_next(&__dpu_it))

/**
 * @brief Intenal macro for DPU_RANK_FOREACH with rank index.
 * @private
 * @hideinitializer
 */
#define _DPU_FOREACH_I(set, dpu, i)                                            \
  for (struct dpu_set_dpu_iterator_t __dpu_it =                                \
           dpu_set_dpu_iterator_from(&set);                                    \
       i = __dpu_it.count, dpu = __dpu_it.next, __dpu_it.has_next;             \
       dpu_set_dpu_iterator_next(&__dpu_it))

/**
 * @brief Iterator on the DPU ranks of a DPU set.
 * @private
 *
 * Mainly used in `DPU_RANK_FOREACH` and `DPU_FOREACH`.
 */
struct dpu_set_rank_iterator_t {
  struct dpu_set_t *set;
  uint32_t count;
  uint32_t next_idx;
  bool has_next;
  struct dpu_set_t next;
};

/**
 * @brief Iterator on the DPUs of a DPU set.
 * @private
 *
 * Mainly used in `DPU_FOREACH`.
 */
struct dpu_set_dpu_iterator_t {
  struct dpu_set_rank_iterator_t rank_iterator;
  uint32_t count;
  uint32_t next_idx;
  bool has_next;
  struct dpu_set_t next;
};

/**
 * @brief Create a DPU rank iterator from the given set.
 * @private
 *
 * Mainly used in `DPU_RANK_FOREACH` and `DPU_FOREACH`.
 *
 * @param set the DPU set we want to iterate on
 * @return The iterator placed on the first element of the set.
 */
struct dpu_set_rank_iterator_t
dpu_set_rank_iterator_from(struct dpu_set_t *set);

/**
 * @brief Advance the iterator to the next element.
 * @private
 *
 * Mainly used in `DPU_RANK_FOREACH` and `DPU_FOREACH`.
 *
 * @param iterator the DPU rank iterator
 */
void dpu_set_rank_iterator_next(struct dpu_set_rank_iterator_t *iterator);

/**
 * @brief Create a DPU iterator from the given set.
 * @private
 *
 * Mainly used in `DPU_FOREACH`.
 *
 * @param set the DPU set we want to iterate on
 * @return The iterator placed on the first element of the set.
 */
struct dpu_set_dpu_iterator_t dpu_set_dpu_iterator_from(struct dpu_set_t *set);

/**
 * @brief Advance the iterator to the next element.
 * @private
 *
 * Mainly used in `DPU_FOREACH`.
 *
 * @param iterator the DPU iterator
 */
void dpu_set_dpu_iterator_next(struct dpu_set_dpu_iterator_t *iterator);

/**
 * @brief Load a program from the memory in all the DPUs of a DPU set.
 *
 * @param dpu_set the targeted DPU set.
 * @param buffer the buffer where the program is stored.
 * @param buffer_size the size of the buffer
 * @param program the DPU program information. Can be `NULL`.
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_load_from_memory(struct dpu_set_t dpu_set, uint8_t *buffer,
                                 size_t buffer_size,
                                 struct dpu_program_t **program);

/**
 * @brief Load a program from a "struct dpu_incbin" in all the DPUs of a DPU
 * set.
 *
 * @param dpu_set the targeted DPU set.
 * @param incbin the struct created with DPU_INCBIN
 * @param program the DPU program information. Can be `NULL`.
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_load_from_incbin(struct dpu_set_t dpu_set,
                                 struct dpu_incbin_t *incbin,
                                 struct dpu_program_t **program);

/**
 * @brief Load a program in all the DPUs of a DPU set.
 *
 * @param dpu_set the targeted DPU set
 * @param binary_path the path of the binary file we want to load in the DPUs
 * @param program the DPU program information. Can be `NULL`.
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_load(struct dpu_set_t dpu_set, const char *binary_path,
                     struct dpu_program_t **program);

/**
 * @brief Get the requested symbol information.
 * @param program the DPU program information
 * @param symbol_name the name of the symbol to look for
 * @param symbol where to store the symbol information if found
 * @return Whether the symbol was found.
 */
dpu_error_t dpu_get_symbol(struct dpu_program_t *program,
                           const char *symbol_name,
                           struct dpu_symbol_t *symbol);

/**
 * @brief Request the boot of all the DPUs in a DPU set.
 * @param dpu_set the identifier of the DPU set we want to boot
 * @param policy how to synchronize with the booted DPUs
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_launch(struct dpu_set_t dpu_set, dpu_launch_policy_t policy);

/**
 * @brief Fetch the current state of the DPU set.
 * @param dpu_set the identifier of the DPU set
 * @param done whether all DPUs of the DPU set have finished
 * @param fault whether any DPU of the DPU set is in fault
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_status(struct dpu_set_t dpu_set, bool *done, bool *fault);

/**
 * @brief Wait for the end of the execution on the DPU set.
 * @param dpu_set the identifier of the DPU set
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_sync(struct dpu_set_t dpu_set);

/**
 * @brief Copy data from the Host memory buffer to one the DPU memories.
 * @param dpu_set the identifier of the DPU set
 * @param symbol_name the name of the DPU symbol where to copy the data
 * @param symbol_offset the byte offset from the base DPU symbol address where
 * to copy the data
 * @param src the host buffer containing the data to copy
 * @param length the number of bytes to copy
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_copy_to(struct dpu_set_t dpu_set, const char *symbol_name,
                        uint32_t symbol_offset, const void *src, size_t length);

/**
 * @brief Copy data from one of the DPU memories to the Host memory buffer.
 * @param dpu_set the identifier of the DPU set
 * @param symbol_name the name of the DPU symbol from where to copy the data
 * @param symbol_offset the byte offset from the base DPU symbol address from
 * where to copy the data
 * @param dst the host buffer where the data is copied
 * @param length the number of bytes to copy
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_copy_from(struct dpu_set_t dpu_set, const char *symbol_name,
                          uint32_t symbol_offset, void *dst, size_t length);

/**
 * @brief Copy data from the Host memory buffer to one the DPU memories.
 * @param dpu_set the identifier of the DPU set
 * @param symbol the DPU symbol where the data is copied
 * @param symbol_offset the byte offset from the base DPU symbol address where
 * to copy the data
 * @param src the host buffer containing the data to copy
 * @param length the number of bytes to copy
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_copy_to_symbol(struct dpu_set_t dpu_set,
                               struct dpu_symbol_t symbol,
                               uint32_t symbol_offset, const void *src,
                               size_t length);

/**
 * @brief Copy data from one of the DPU memories to the Host memory buffer.
 * @param dpu_set the identifier of the DPU set
 * @param symbol the DPU symbol from where the data is copied
 * @param symbol_offset the byte offset from the base DPU symbol address from
 * where to copy the data
 * @param dst the host buffer where the data is copied
 * @param length the number of bytes to copy
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_copy_from_symbol(struct dpu_set_t dpu_set,
                                 struct dpu_symbol_t symbol,
                                 uint32_t symbol_offset, void *dst,
                                 size_t length);

/**
 * @brief Set the Host buffer of all DPUs of the DPU set for the next memory
 * transfer.
 *
 * `NULL` can be used to clear the buffer pointer.
 * An error will be reported if any buffer was already set; the buffer pointer
 * will be overridden.
 *
 * @param dpu_set the identifier of the DPU set
 * @param buffer pointer to the host buffer
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_prepare_xfer(struct dpu_set_t dpu_set, void *buffer);

/**
 * @brief Structure that stores the information about each scatter transfer
 * memory block
 */
struct sg_block_info {
  /** Starting address of the block */
  uint8_t *addr;
  /** Number of bytes to transfer for this block */
  uint32_t length;
};

/**
 * @brief User API prototype of the get_block function for scatter transfers
 * @param out (output) block information
 * @param dpu_index index of the dpu
 * @param block_index index of the block for the current DPU
 * @param args user arguments
 * @return Whether this block exists or not
 */
typedef bool (*get_block_func_t)(struct sg_block_info *out, uint32_t dpu_index,
                                 uint32_t block_index, void *args);

/**
 * @brief User API structure that stores the scatter transfer get_block function
 * pointer and its arguments, provided by user application
 */
typedef struct get_block_t {
  /** The get_block function */
  get_block_func_t f;
  /** User arguments for the get_block function */
  void *args;
  /** Size of the user arguments */
  size_t args_size;
} get_block_t;

/**
 * @brief Execute the scatter/gather memory transfer on the DPU set
 *
 * First, prepare the host buffer blocks to be gathered from the HOST memory
 * to the DPU MRAM or to be scattered from the DPU MRAM to the HOST memory,
 * depending on the transfer direction. Then, performs the transfer.
 * Blocks are prepared using the callback function (get_block_info.f),
 * defined within the user application.
 *
 * @param dpu_set the identifier of the DPU set
 * @param xfer direction of the transfer
 * @param symbol_name the name of the DPU symbol where the transfer starts
 * @param symbol_offset the byte offset from the base DPU symbol address where
 * the transfer starts
 * @param length the number of bytes to copy
 * @param get_block_info a structure containing user function and user
 * arguments, used to prepare the transfer
 * @param flags options of the transfer
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_push_sg_xfer(struct dpu_set_t dpu_set, dpu_xfer_t xfer,
                             const char *symbol_name, uint32_t symbol_offset,
                             size_t length, get_block_t *get_block_info,
                             dpu_sg_xfer_flags_t flags);

/**
 * @brief Execute the scatter/gather memory transfer on the DPU set
 *
 * First, prepare the host buffer blocks to be gathered from the HOST memory
 * to the DPU MRAM or to be scattered from the DPU MRAM to the HOST memory,
 * depending on the transfer direction. Then, performs the transfer.
 * Blocks are prepared using the callback function (get_block_info.f),
 * defined within the user application.
 *
 * @param dpu_set the identifier of the DPU set
 * @param xfer direction of the transfer
 * @param symbol the DPU symbol where the transfer starts
 * @param symbol_offset the byte offset from the base DPU symbol address where
 * the transfer starts
 * @param length the number of bytes to copy
 * @param get_block_info a structure containing user function and user
 * arguments, used to prepare the transfer
 * @param flags options of the transfer
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_push_sg_xfer_symbol(struct dpu_set_t dpu_set, dpu_xfer_t xfer,
                                    struct dpu_symbol_t symbol,
                                    uint32_t symbol_offset, size_t length,
                                    get_block_t *get_block_info,
                                    dpu_sg_xfer_flags_t flags);

/**
 * @brief Execute the memory transfer on the DPU set
 *
 * Use the host buffers previously defined by `dpu_prepare_xfer`.
 * When reading memory from the DPUs, if a host buffer is used for multiple
 * DPUs, no error will be reported, and the buffer contents are undefined.
 *
 * @param dpu_set the identifier of the DPU set
 * @param xfer direction of the transfer
 * @param symbol_name the name of the DPU symbol where the transfer starts
 * @param symbol_offset the byte offset from the base DPU symbol address where
 * the transfer starts
 * @param length the number of bytes to copy
 * @param flags options of the transfer
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_push_xfer(struct dpu_set_t dpu_set, dpu_xfer_t xfer,
                          const char *symbol_name, uint32_t symbol_offset,
                          size_t length, dpu_xfer_flags_t flags);

/**
 * @brief Execute the memory transfer on the DPU set
 *
 * Use the host buffers previously defined by `dpu_prepare_xfer`.
 * When reading memory from the DPUs, if a host buffer is used for multiple
 * DPUs, no error will be reported, and the buffer contents are undefined.
 *
 * @param dpu_set the identifier of the DPU set
 * @param xfer direction of the transfer
 * @param symbol the DPU symbol where the transfer starts
 * @param symbol_offset the byte offset from the base DPU symbol address where
 * the transfer starts
 * @param length the number of bytes to copy
 * @param flags options of the transfer
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_push_xfer_symbol(struct dpu_set_t dpu_set, dpu_xfer_t xfer,
                                 struct dpu_symbol_t symbol,
                                 uint32_t symbol_offset, size_t length,
                                 dpu_xfer_flags_t flags);

/**
 * @brief Execute the broadcast memory transfer on the DPU set
 *
 * @param dpu_set the identifier of the DPU set
 * @param symbol_name the name of the DPU symbol where the transfer starts
 * @param symbol_offset the byte offset from the base DPU symbol address where
 * the transfer starts
 * @param src the host buffer containing the data to copy
 * @param length the number of bytes to copy
 * @param flags options of the transfer
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_broadcast_to(struct dpu_set_t dpu_set, const char *symbol_name,
                             uint32_t symbol_offset, const void *src,
                             size_t length, dpu_xfer_flags_t flags);

/**
 * @brief Execute the broadcast memory transfer on the DPU set
 *
 * @param dpu_set the identifier of the DPU set
 * @param symbol the DPU symbol where the transfer starts
 * @param symbol_offset the byte offset from the base DPU symbol address where
 * the transfer starts
 * @param src the host buffer containing the data to copy
 * @param length the number of bytes to copy
 * @param flags options of the transfer
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_broadcast_to_symbol(struct dpu_set_t dpu_set,
                                    struct dpu_symbol_t symbol,
                                    uint32_t symbol_offset, const void *src,
                                    size_t length, dpu_xfer_flags_t flags);

/**
 * @brief Execute a user-defined function with the given DPU set as argument
 *
 * @param dpu_set the identifier of the DPU set
 * @param callback the function to call
 * @param args the argument to pass to the callback function
 * @param flags options of the callback
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_callback(struct dpu_set_t dpu_set,
                         dpu_error_t (*callback)(struct dpu_set_t, uint32_t,
                                                 void *),
                         void *args, dpu_callback_flags_t flags);

/**
 * @brief Embed a binary inside the ELF program.
 * @param name the name of the "struct dpu_incbin" that will be created by the
 * MACRO.
 * @param file the binary to embbed.
 * @hideinitializer
 */
/* clang-format off */
#define DPU_INCBIN(name, file)                                                     \
    __asm__(".pushsection .rodata.dpu_incbin\n"                                    \
            ".global dpu_incbin_" _STR(name) "\n"                                  \
            ".type dpu_incbin_" _STR(name) ", @object\n"                           \
            "dpu_incbin_" _STR(name) ":\n"                                         \
            ".incbin \"" file "\"\n"                                               \
                                                                                   \
            ".size dpu_incbin_" _STR(name) ", .-dpu_incbin_" _STR(name) "\n"       \
            ".global dpu_incbin_" _STR(name) "_size\n"                             \
            "dpu_incbin_"  _STR(name) "_size = . - dpu_incbin_" _STR(name) "\n"    \
            ".popsection\n"                                                        \
    );                                                                             \
    extern const void *dpu_incbin_ ## name;                                        \
    extern const void *dpu_incbin_ ## name ## _size;                               \
    struct dpu_incbin_t name = {.buffer = (uint8_t *)&dpu_incbin_ ## name ,        \
                                .size = (size_t)&dpu_incbin_ ## name ## _size,     \
                                .path = file };
/* clang-format on */

/**
 * @brief reads and displays the contents of the log of a DPU
 * @param set the dpu_set_t from which to extract the log (should be a single
 * DPU)
 * @param stream output stream where messages should be sent
 * @return whether the log was successfully read
 */
dpu_error_t dpu_log_read(struct dpu_set_t set, FILE *stream);

#endif // DPU_H
