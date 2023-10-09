/* Copyright 2020 UPMEM. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef DPU_CHECKPOINT_H
#define DPU_CHECKPOINT_H

#include "dpu_error.h"
#include "dpu_types.h"

/**
 * @file dpu_checkpoint.h
 * @brief C API to create and manage DPU checkpoints.
 */

/**
 * @brief Options for a DPU checkpoint.
 */
typedef enum _dpu_checkpoint_flags_t {
  DPU_CHECKPOINT_NONE = 0x0,
  /** Saves the internal state of the DPU. */
  DPU_CHECKPOINT_INTERNAL = 0x1,
  /** Saves the Instruction RAM of the DPU. */
  DPU_CHECKPOINT_IRAM = 0x2,
  /** Saves the Main RAM of the DPU. */
  DPU_CHECKPOINT_MRAM = 0x8,
  /** Saves the Work RAM of the DPU. */
  DPU_CHECKPOINT_WRAM = 0x10,
} dpu_checkpoint_flags_t;

/**
 * @brief Option for a DPU checkpoint to save the whole DPU state.
 * @hideinitializer
 */
#define DPU_CHECKPOINT_ALL                                                     \
  (DPU_CHECKPOINT_INTERNAL | DPU_CHECKPOINT_IRAM | DPU_CHECKPOINT_MRAM |       \
   DPU_CHECKPOINT_WRAM)

/**
 * @brief Extracts the DPU state.
 * @param set the identifier of the DPU set (must be a single DPU)
 * @param flags options on what to extract from the DPU state
 * @param context storage for the DPU state
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_checkpoint_save(struct dpu_set_t set,
                                dpu_checkpoint_flags_t flags,
                                struct dpu_context_t *context);

/**
 * @brief Restores a DPU state.
 * @param set the identifier of the DPU set (must be a single DPU)
 * @param flags options on what to restore from the DPU state
 * @param context the DPU state to restore
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_checkpoint_restore(struct dpu_set_t set,
                                   dpu_checkpoint_flags_t flags,
                                   struct dpu_context_t *context);

/**
 * @brief Computes the size of the context once serialized.
 * @param context the DPU state to serialize
 * @return The serialized context size.
 */
uint32_t
dpu_checkpoint_get_serialized_context_size(struct dpu_context_t *context);

/**
 * @brief Serializes a DPU state.
 * @param context the DPU state to serialize
 * @param serialized_context storage for the DPU state content
 * @param serialized_context_size storage for the number of bytes of the DPU
 * state content
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_checkpoint_serialize(struct dpu_context_t *context,
                                     uint8_t **serialized_context,
                                     uint32_t *serialized_context_size);

/**
 * @brief Deserializes a DPU state.
 * @param serialized_context the serialized DPU state
 * @param serialized_context_size the number of bytes of the serialized DPU
 * state
 * @param context storage for the DPU state
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_checkpoint_deserialize(const uint8_t *serialized_context,
                                       uint32_t serialized_context_size,
                                       struct dpu_context_t *context);

/**
 * @brief Free a DPU context allocated by dpu_checkpoint.
 * @param context the DPU context to free
 * @return Whether the operation was successful.
 */
dpu_error_t dpu_checkpoint_free(struct dpu_context_t *context);

#endif /* DPU_CHECKPOINT_H */
