#pragma once

#include "brt/core/framework/op_kernel.h"

namespace brt {
namespace pim {
namespace hbmpim {

/**
 * GEMV Ops
 * This is just an example for OpKernel.
 * All elementwise ops should be generated through macro or generator.
 */
template <typename T> class GEMV final : public OpKernel {
public:
  explicit GEMV(const OpKernelInfo &info);

  common::Status RunImpl(const ExecutionContext &) override;
  common::Status ProloguePerSession() override;

  common::Status ProloguePerFrame(const ExecutionContext &) override;
};

} // namespace hbmpim
} // namespace pim
} // namespace brt