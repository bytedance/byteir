/* Copyright 2020 UPMEM. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef DPU_ERROR_H
#define DPU_ERROR_H

/**
 * @file dpu_error.h
 * @brief Define the possible status returned by the API functions.
 */

/**
 * @brief Status returned by any API operation to specify the success or failure of the operation.
 */
typedef enum dpu_error_t {
    /** The operation was successful. */
    DPU_OK,
    /** Error returned when there is an unexpected error in the API. */
    DPU_ERR_INTERNAL,
    /** Error returned when a system function (eg. malloc) failed. */
    DPU_ERR_SYSTEM,
    /** Error returned when there is an unexpected error in the rank driver. */
    DPU_ERR_DRIVER,
    /** Error returned when there is an error during the rank allocation. */
    DPU_ERR_ALLOCATION,
    /** Error returned when trying to do an invalid operation on a DPU set. */
    DPU_ERR_INVALID_DPU_SET,
    /** Error returned when the specified thread id is invalid. */
    DPU_ERR_INVALID_THREAD_ID,
    /** Error returned when the specified notify id is invalid. */
    DPU_ERR_INVALID_NOTIFY_ID,
    /** Error returned when the WRAM access parameters are invalid. */
    DPU_ERR_INVALID_WRAM_ACCESS,
    /** Error returned when the IRAM access parameters are invalid. */
    DPU_ERR_INVALID_IRAM_ACCESS,
    /** Error returned when the MRAM access parameters are invalid. */
    DPU_ERR_INVALID_MRAM_ACCESS,
    /** Error returned when the symbol access parameters are invalid. */
    DPU_ERR_INVALID_SYMBOL_ACCESS,
    /** Error returned when the DPU MRAM is not accessible to the host. */
    DPU_ERR_MRAM_BUSY,
    /** Error returned when trying to prepare a memory transfer on an already prepared dpu. */
    DPU_ERR_TRANSFER_ALREADY_SET,
    /** Error returned when no program has been loaded into the dpu. */
    DPU_ERR_NO_PROGRAM_LOADED,
    /** Error returned when fetching a DPU symbol on DPUs with different programs */
    DPU_ERR_DIFFERENT_DPU_PROGRAMS,
    /** Error returned when the DPU MRAM is corrupted and cannot be repaired. */
    DPU_ERR_CORRUPTED_MEMORY,
    /** Error returned when accessing a disabled DPU. */
    DPU_ERR_DPU_DISABLED,
    /** Error returned when trying to boot an already running DPU. */
    DPU_ERR_DPU_ALREADY_RUNNING,
    /** Error returned when the specified memory transfer is unknown. */
    DPU_ERR_INVALID_MEMORY_TRANSFER,
    /** Error returned when the specified launch policy is unknown. */
    DPU_ERR_INVALID_LAUNCH_POLICY,
    /** Error returned when the DPU is in fault. */
    DPU_ERR_DPU_FAULT,
    /** Error returned when the ELF file is invalid. */
    DPU_ERR_ELF_INVALID_FILE,
    /** Error returned when the ELF file is not found. */
    DPU_ERR_ELF_NO_SUCH_FILE,
    /** Error returned when the section is not found in the ELF file. */
    DPU_ERR_ELF_NO_SUCH_SECTION,
    /** Error returned when the operation timed out. */
    DPU_ERR_TIMEOUT,
    /** Error returned when the DPU profile passed during allocation is invalid. */
    DPU_ERR_INVALID_PROFILE,
    /** Error returned when fetching an undefined symbol in a DPU program. */
    DPU_ERR_UNKNOWN_SYMBOL,
    /** Error returned when the parsing of the log buffer is not what we expect. */
    DPU_ERR_LOG_FORMAT,
    /** Error returned when we cannot find log information in the dpu structure. */
    DPU_ERR_LOG_CONTEXT_MISSING,
    /** Error returned when the log buffer was too small to contain all messages */
    DPU_ERR_LOG_BUFFER_TOO_SMALL,
    /** Error returned when the VPD file is invalid. */
    DPU_ERR_VPD_INVALID_FILE,
    /** Error returned when SRAM repairs have not been generated. */
    DPU_ERR_VPD_NO_REPAIR,
    /** Error returned when trying to do a multiple-rank operation with 0 thread per rank */
    DPU_ERR_NO_THREAD_PER_RANK,
    /** Error returned when trying to serialize a ctx in a buffer with an incorrect size */
    DPU_ERR_INVALID_BUFFER_SIZE,
    /** Error returned when enqueueing a non-blocking synchronous callback */
    DPU_ERR_NONBLOCKING_SYNC_CALLBACK,
    /** Error returned when the DPU program uses more tasklets than the hardware has threads */
    DPU_ERR_TOO_MANY_TASKLETS,
    /** Error returned when the number of scatter gather transfer blocks exceeds the maximal number of blocks */
    DPU_ERR_SG_TOO_MANY_BLOCKS,
    /** Error returned when the scatter transfer total number of bytes in prepared blocks mismatch with
        the size of the transfer (dpu_push_sg_xfer(..,.length,...)) */
    DPU_ERR_SG_LENGTH_MISMATCH,
    /** Error returned when a scatter gather transfer is performed without prior SG transfer activation */
    DPU_ERR_SG_NOT_ACTIVATED,
    /** Error returned when scatter gather transfer symbol is not a MRAM symbol */
    DPU_ERR_SG_NOT_MRAM_SYMBOL,
    /** Error returned if one sg xfer is launched and the sg buffer pool has not been initialized */
    DPU_ERR_ASYNC_JOBS = 1U << 31,
} dpu_error_t;

/**
 * @brief Transform a dpu_error_t into a string.
 * @param status the api status to stringify
 * @return The string associated to the specified status. It is the user responsability to free the returned string when it is not
 * needed anymore.
 */
const char *
dpu_error_to_string(dpu_error_t status);

/**
 * @brief In an error from an async job, offset of the error value where the job type information can be found.
 * @private
 */
#define DPU_ERROR_ASYNC_JOB_TYPE_SHIFT (16)

#endif // DPU_ERROR_H
