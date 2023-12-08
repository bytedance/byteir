// Copyright (c) Megvii Inc.
// Licensed under Apache License, Version 2.0
// ===========================================================================
// Modification Copyright 2022 ByteDance Ltd. and/or its affiliates.

#pragma once

#include <string>

namespace brt {

//  DContext is an abstraction of communication contexts (e.g. cuda stream)
//  on different platforms, a context should be passed as a parameter when
//  a communicator operation is called
class DContext {
public:
  virtual std::string type() const = 0;
  virtual ~DContext() = default;
};

} // namespace brt