/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- OFCompilerTypes.cpp ------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#pragma once

namespace onnx_frontend {

/* Type of compiler emission targets */
typedef enum {
  EmitONNXIR,
  EmitMhloIR,
} EmissionTargetType;

/* Input IR can be at one of these levels */
typedef enum {
  ONNXLevel,
  MhloLevel,
} InputIRLevelType;

} // namespace onnx_frontend
