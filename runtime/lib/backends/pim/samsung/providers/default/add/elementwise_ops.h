#pragma once

#include "brt/core/framework/op_kernel.h"




namespace brt {
namespace pim {
namespace hbm {

/**
 * Add Ops
 * This is just an example for OpKernel.
 * All elementwise ops should be generated through macro or generator.
 */
template <typename T> class Add final : public OpKernel {
public:
  explicit Add(const OpKernelInfo &info) ;

  common::Status RunImpl(const ExecutionContext &) override;
    common::Status ProloguePerSession() override;

  common::Status ProloguePerFrame(const ExecutionContext &) override;
};

} // namespace hbm
} // namespace pim
} // namespace brt