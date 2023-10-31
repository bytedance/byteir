
#pragma once

#include "brt/core/common/status.h"

// #include "pim.h"

namespace brt {
namespace pim {
namespace hbm {
// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------
enum hbm_error_t{
  DPU_OK = 0,
  DPU_ERR_SYSTEM
};


template <typename ERRTYPE, bool THRW>
[[nodiscard]] std::conditional_t<THRW, void, common::Status>
HBMCall(ERRTYPE retCode, const char *exprString, const char *libName,
        ERRTYPE successCode, const char *msg = "");
} // namespace hbm
} // namespace pim
} // namespace brt

#define BRT_HBM_CALL(expr)                                                     \
  (::brt::pim::hbm::HBMCall<hbm_error_t, false>((expr), #expr, "HBM", DPU_OK))

#define BRT_HBM_CALL_THRW(expr)                                                \
  (::brt::pim::hbm::HBMCall<hbm_error_t, true>((expr), #expr, "HBM", DPU_OK))
