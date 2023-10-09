/* Copyright 2020 UPMEM. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef __DPU_TARGET_H
#define __DPU_TARGET_H

/**
 * @file dpu_target.h
 * @brief Definition of the different DPU backend types.
 */

#include <stdbool.h>
#include <string.h>

/**
 * @brief The different DPU backend types.
 */
typedef enum _dpu_type_t {
    FUNCTIONAL_SIMULATOR = 0,
    CYCLE_ACCURATE_SIMULATOR = 1,
    RTL_SIMULATOR = 2,
    HW = 3,
    BACKUP_SPI = 4,
    SCENARIO = 5,

    NB_OF_DPU_TYPES = 6
} dpu_type_t;

/**
 * @brief Get the display string for a given DPU type.
 * @param dpu_type the DPU type to stringify
 * @return The string associated to the DPU type.
 */
static inline const char *
dpu_type_to_string(dpu_type_t dpu_type)
{
    switch (dpu_type) {
        case FUNCTIONAL_SIMULATOR:
            return "FUNCTIONAL_SIMULATOR";
        case CYCLE_ACCURATE_SIMULATOR:
            return "CYCLE_ACCURATE_SIMULATOR";
        case RTL_SIMULATOR:
            return "RTL_SIMULATOR";
        case HW:
            return "HW";
        case BACKUP_SPI:
            return "BACKUP_SPI";
        case SCENARIO:
            return "SCENARIO";
        default:
            return "UNKNOWN";
    }
}

/**
 * @brief Get the DPU type from a string given by the UPMEM_PROFILE.
 * @param string the string to parse
 * @param dpu_type storage for the parsed DPU type
 * @return Whether the string could be parsed into a valid DPU type.
 */
static inline bool
dpu_type_from_profile_string(const char *string, dpu_type_t *dpu_type)
{
    if (strcmp(string, "simulator") == 0) {
        *dpu_type = FUNCTIONAL_SIMULATOR;
        return true;
    }
    if (strcmp(string, "cas") == 0) {
        *dpu_type = CYCLE_ACCURATE_SIMULATOR;
        return true;
    }
    if (strcmp(string, "model") == 0) {
        *dpu_type = RTL_SIMULATOR;
        return true;
    }
    if (strcmp(string, "hw") == 0) {
        *dpu_type = HW;
        return true;
    }
    if (strcmp(string, "spi") == 0) {
        *dpu_type = BACKUP_SPI;
        return true;
    }
    if (strcmp(string, "scenario") == 0) {
        *dpu_type = SCENARIO;
        return true;
    }
    return false;
}

#endif /* __DPU_TARGET_H */
