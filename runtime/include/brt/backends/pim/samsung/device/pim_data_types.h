/*
 * Copyright (C) 2022 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose only)
 */

#ifndef _PIM_DATA_TYPE_H_
#define _PIM_DATA_TYPE_H_

#include <stddef.h>
#include <stdint.h>
#include <unordered_map>
#include <vector>

#define __PIM_API__

typedef enum __PimRuntimeType {
    RT_TYPE_HIP,
    RT_TYPE_OPENCL,
} PimRuntimeType;

typedef enum __PimDevice {
    GPU,
} PimDevice;

typedef enum __PimMemType {
    MEM_TYPE_HOST,
    MEM_TYPE_DEVICE,
    MEM_TYPE_PIM,
} PimMemType;

typedef enum __PimMemFlag {
    ELT_OP,
    GEMV_INPUT,
    GEMV_WEIGHT,
    GEMV_OUTPUT,
    GEMM_INPUT,
    GEMM_WEIGHT,
    GEMM_BIAS,
    GEMM_OUTPUT,
} PimMemFlag;

typedef enum __PimMemCpyType {
    HOST_TO_HOST,
    HOST_TO_DEVICE,
    HOST_TO_PIM,
    DEVICE_TO_HOST,
    DEVICE_TO_DEVICE,
    DEVICE_TO_PIM,
    PIM_TO_HOST,
    PIM_TO_DEVICE,
    PIM_TO_PIM,
} PimMemCpyType;

typedef enum __PimOpType {
    OP_GEMV,
    OP_GEMM,
    OP_ELT_ADD,
    OP_ELT_MUL,
    OP_RELU,
    OP_BN,
    OP_COPY,
    OP_DUMMY,
} PimOpType;

typedef enum __PimGemmOrder {
    W_X_I,
    I_X_W,
} PimGemmOrder;

typedef enum __PimActivationFunction {
    NONE,
    ACT_RELU,
} PimActFunc;

typedef enum __PimPrecision {
    PIM_FP16,
    PIM_INT8,
} PimPrecision;

typedef enum __PimDataLayoutType {
    RAW,
    CHWISE_GEMM_WEIGHT,
    ALIGNED_GEMM_WEIGHT,
} PimDataLayoutType;

typedef struct __PimBufferShape {
    uint32_t n;
    uint32_t c;
    uint32_t h;
    uint32_t w;
} PimBShape;

typedef struct __PimBufferObject {
    PimMemType mem_type;
    PimDataLayoutType data_layout_type;
    PimBShape bshape;
    PimBShape bshape_r;
    PimPrecision precision;
    size_t size;
    size_t size_r;
    void *data;
    bool use_user_ptr;
    bool transposed;
} PimBo;

#if PIM_COMPILER_ENABLE == 1
typedef struct __PimTarget {
    PimRuntimeType runtime;
    PimDevice device;
    PimPrecision precision;
} PimTarget;

typedef struct __PimCompiledObject {
    int32_t return_val;
    PimBo *output_pimbo;
    std::vector<PimBo *> input_pimbo;
    std::vector<PimBo *> new_pimbo;
    std::string kernel;
    std::string crf_binary;
    uint32_t num_blocks;
    uint32_t num_threads;
    std::vector<std::string> op_order;
    std::unordered_map<std::string, PimBo *> pimbo_map;
} PimCompiledObj;
#endif

typedef struct __PimGemmDescriptor {
    PimBShape in_bshape;
    PimBShape in_bshape_r;
    PimBShape wei_bshape;
    PimBShape wei_bshape_r;
    PimBShape bias_bshape;
    PimBShape bias_bshape_r;
    PimBShape out_bshape;
    PimBShape out_bshape_r;
    PimPrecision precision;
    PimGemmOrder gemm_order;
} PimGemmDesc;

typedef struct __PimDescriptor {
    PimBShape bshape;
    PimBShape bshape_r;
    PimPrecision precision;
    PimOpType op_type;
} PimDesc;

typedef struct __PimCopy3D {
    /* Source information */
    size_t src_x_in_bytes, src_y, src_z; /* X, Y, Z offset of the src pointer */
    PimMemType src_mem_type;             /* Memory type of the source memory */
    const void *src_ptr;                 /* Source pointer; ignored if srcBo != nullptr */
    size_t src_pitch;                    /* Source row width in bytes; ignored if srcBo != nullptr */
    size_t src_height;                   /* Source height (scalar); ignored if srcBo != nullptr */
    const PimBo *src_bo;                 /* Source PIM buffer object */
    /* Destination information */
    size_t dst_x_in_bytes, dst_y, dst_z; /* X, Y, Z offset of the src pointer */
    PimMemType dst_mem_type;             /* Memory type of the destination memory */
    void *dst_ptr;                       /* Destination pointer; ignored if dstBo != nullptr */
    size_t dst_pitch;                    /* Destination row width in bytes; ignored if dstBo != null */
    size_t dst_height;                   /* Destination height (scalar); ignored if dstBo != nullptr */
    PimBo *dst_bo;                       /* Destination PIM buffer object */
    /* Slice information */
    size_t width_in_bytes; /* Width of the slice to copy in bytes */
    size_t height;         /* Height of the slice to copy (scalar) */
    size_t depth;          /* Depth of the slice to copy (scalar) */
} PimCopy3D;

#endif /* _PIM_DATA_TYPE_H_ */
