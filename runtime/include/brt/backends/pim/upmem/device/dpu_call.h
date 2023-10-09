
#pragma once

#include "brt/core/common/status.h"

#include "./dpu_types.h"

namespace brt {
namespace pim {
namespace upmem {
// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE, bool THRW>
[[nodiscard]] std::conditional_t<THRW, void, common::Status>
UPMEMCall(ERRTYPE retCode, const char *exprString, const char *libName,
          ERRTYPE successCode, const char *msg = "");
} // namespace upmem
} // namespace pim
} // namespace brt

#define BRT_UPMEM_CALL(expr)                                                   \
  (::brt::pim::upmem::UPMEMCall<dpu_error_t, false>((expr), #expr, "UPMEM",    \
                                                    DPU_OK))

#define BRT_UPMEM_CALL_THRW(expr)                                              \
  (::brt::pim::upmem::UPMEMCall<dpu_error_t, true>((expr), #expr, "UPMEM",     \
                                                   DPU_OK))
