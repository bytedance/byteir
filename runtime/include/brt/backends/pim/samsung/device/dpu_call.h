
#pragma once

#include "brt/core/common/status.h"

// #include "pim.h"

namespace brt {
namespace pim {
namespace hbmpim {
// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------
enum hbm_error_t{
  DPU_OK = 0,
  DPU_ERR_SYSTEM
};


template <typename ERRTYPE, bool THRW>
[[nodiscard]] std::conditional_t<THRW, void, common::Status>
HBMPIMCall(ERRTYPE retCode, const char *exprString, const char *libName,
        ERRTYPE successCode, const char *msg = "");
} // namespace hbmpim
} // namespace pim
} // namespace brt

#define BRT_HBMPIM_CALL(expr)                                                     \
  (::brt::pim::hbmpim::HBMPIMCall<hbm_error_t, false>((expr), #expr, "HBMPIM", DPU_OK))

#define BRT_HBMPIM_CALL_THRW(expr)                                                \
  (::brt::pim::hbmpim::HBMPIMCall<hbm_error_t, true>((expr), #expr, "HBMPIM", DPU_OK))
