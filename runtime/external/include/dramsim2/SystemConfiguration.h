/*********************************************************************************
 *  Copyright (c) 2010-2011, Elliott Cooper-Balis
 *                             Paul Rosenfeld
 *                             Bruce Jacob
 *                             University of Maryland
 *                             dramninjas [at] gmail [dot] com
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright notice,
 *        this list of conditions and the following disclaimer in the documentation
 *        and/or other materials provided with the distribution.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************************/

#ifndef SYSCONFIG_H
#define SYSCONFIG_H

#include <stdint.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#ifdef __APPLE__
#include <sys/types.h>
#endif

#include "PrintMacros.h"

extern std::ofstream cmd_verify_out;  // used by BusPacket.cpp if VERIFICATION_OUTPUT is enabled
// extern std::ofstream visDataOut;

// TODO: namespace these to DRAMSim::
extern bool VERIFICATION_OUTPUT;  // output suitable to feed to modelsim

extern bool DEBUG_TRANS_Q;
extern bool DEBUG_CMD_Q;
extern bool DEBUG_ADDR_MAP;
extern bool DEBUG_BANKSTATE;
extern bool DEBUG_BUS;
extern bool DEBUG_BANKS;
extern bool DEBUG_POWER;
extern bool USE_LOW_POWER;
extern bool VIS_FILE_OUTPUT;
extern bool PRINT_CHAN_STAT;
extern bool DEBUG_PIM_TIME;
extern bool DEBUG_CMD_TRACE;
extern bool DEBUG_PIM_BLOCK;

extern std::string SIM_TRACE_FILE;
extern bool SHOW_SIM_OUTPUT;
extern bool LOG_OUTPUT;

namespace DRAMSim
{
enum TraceType { k6, mase, misc };

enum AddressMappingScheme {
    Scheme1 = 1,
    Scheme2,
    Scheme3,
    Scheme4,
    Scheme5,
    Scheme6,
    Scheme7,
    /* FIXME: need to change at launch shcha */
    Scheme8,
    VegaScheme
};

// used in MemoryController and CommandQueue
enum RowBufferPolicy { OpenPage, ClosePage };

// Only used in CommandQueue
enum QueuingStructure { PerRank, PerRankPerBank };
enum SchedulingPolicy { RankThenBankRoundRobin, BankThenRankRoundRobin };

enum PIMMode { mac_in_bankgroup, mac_in_bank };
enum PIMPrecision { FP16, INT8, FP32 };

enum class dramMode { SB, HAB, HAB_PIM };

enum class pimBankType { EVEN_BANK, ODD_BANK, ALL_BANK };
// set by IniReader.cpp

typedef void (*returnCallBack_t)(unsigned id, uint64_t addr, uint64_t clockcycle);
typedef void (*powerCallBack_t)(double bgpower, double burstpower, double refreshpower, double actprepower);

};  // namespace DRAMSim

/* SystemConfiguration Singletone Class */

#include "ConfigurationDB.h"
#include "ConfigurationData.h"

using namespace std;

namespace DRAMSim
{
template <typename T>
class SystemConfigurationBase
{
   public:
    typedef T dataType;
    virtual T getValue(const string& key) = 0;
    void setValue(const string& key, const string& value, const paramType& ptype)
    {
        ConfigurationDB& _ptrDB = ConfigurationDB::getDB();
        ConfigurationData newValue = {key, getVarType(), ptype, value};
        _ptrDB.update(newValue);
    }

   private:
    varType getVarType()
    {
        if (is_same<dataType, string>::value) {
            return STRING;
        } else if (is_same<dataType, unsigned>::value) {
            return UINT;
        } else if (is_same<dataType, uint64_t>::value) {
            return UINT64;
        } else if (is_same<dataType, float>::value) {
            return FLOAT;
        } else if (is_same<dataType, bool>::value) {
            return BOOL;
        } else {
            throw invalid_argument("unknown variable type");
        }
    }
};

class StringSystemConfiguration : public SystemConfigurationBase<string>
{
   public:
    virtual string getValue(const string& key) override
    {
        ConfigurationDB& _ptrDB = ConfigurationDB::getDB();
        const ConfigurationData* ptrConf = _ptrDB.find(key);
        return (ptrConf != nullptr) ? ptrConf->value : 0;
    }
};

class UnsignedSystemConfiguration : public SystemConfigurationBase<unsigned>
{
   public:
    virtual unsigned getValue(const string& key) override
    {
        ConfigurationDB& _ptrDB = ConfigurationDB::getDB();
        const ConfigurationData* ptrConf = _ptrDB.find(key);
        return (ptrConf != nullptr) ? static_cast<unsigned>(stoul(ptrConf->value)) : 0;
    }
};

class Uint64SystemConfiguration : public SystemConfigurationBase<uint64_t>
{
   public:
    virtual uint64_t getValue(const string& key) override
    {
        ConfigurationDB& _ptrDB = ConfigurationDB::getDB();
        const ConfigurationData* ptrConf = _ptrDB.find(key);
        return (ptrConf != nullptr) ? stoul(ptrConf->value) : 0;
    }
};

class FloatSystemConfiguration : public SystemConfigurationBase<float>
{
   public:
    virtual float getValue(const string& key) override
    {
        ConfigurationDB& _ptrDB = ConfigurationDB::getDB();
        const ConfigurationData* ptrConf = _ptrDB.find(key);
        return (ptrConf != nullptr) ? stof(ptrConf->value) : 0;
    }
};

class BoolSystemConfiguration : public SystemConfigurationBase<bool>
{
   public:
    virtual bool getValue(const string& key) override
    {
        ConfigurationDB& _ptrDB = ConfigurationDB::getDB();
        const ConfigurationData* ptrConf = _ptrDB.find(key);
        return (ptrConf != nullptr && ptrConf->value == "true") ? true : false;
    }
};

class SystemConfiguration
{
   public:
    static StringSystemConfiguration& getStringSystemConfig()
    {
        static StringSystemConfiguration unique_instance;
        return unique_instance;
    }

    static UnsignedSystemConfiguration& getUnsignedSystemConfig()
    {
        static UnsignedSystemConfiguration unique_instance;
        return unique_instance;
    }

    static Uint64SystemConfiguration& getUint64SystemConfig()
    {
        static Uint64SystemConfiguration unique_instance;
        return unique_instance;
    }

    static FloatSystemConfiguration& getFloatSystemConfig()
    {
        static FloatSystemConfiguration unique_instance;
        return unique_instance;
    }

    static BoolSystemConfiguration& getBoolSystemConfig()
    {
        static BoolSystemConfiguration unique_instance;
        return unique_instance;
    }
};

#define getConfigParam(variableType, key) get##variableType##Config(key)

static inline string getSTRINGConfig(const string& key)
{
    StringSystemConfiguration& config = SystemConfiguration::getStringSystemConfig();
    return config.getValue(key);
}

static inline unsigned getUINTConfig(const string& key)
{
    UnsignedSystemConfiguration& config = SystemConfiguration::getUnsignedSystemConfig();
    return config.getValue(key);
}

static inline uint64_t getUINT64Config(const string& key)
{
    Uint64SystemConfiguration& config = SystemConfiguration::getUint64SystemConfig();
    return config.getValue(key);
}

static inline float getFLOATConfig(const string& key)
{
    FloatSystemConfiguration& config = SystemConfiguration::getFloatSystemConfig();
    return config.getValue(key);
}

static inline bool getBOOLConfig(const string& key)
{
    BoolSystemConfiguration& config = SystemConfiguration::getBoolSystemConfig();
    return config.getValue(key);
}

#define setDevConfigParam(variableType, key, value) set##variableType##Config(key, value, DEV_PARAM)
#define setSysConfigParam(variableType, key, value) set##variableType##Config(key, value, SYS_PARAM)

static inline void setSTRINGConfig(const string& key, const string& value, const paramType& ptype)
{
    StringSystemConfiguration& config = SystemConfiguration::getStringSystemConfig();
    config.setValue(key, value, ptype);
}

static inline void setUINTConfig(const string& key, const unsigned& value, const paramType& ptype)
{
    UnsignedSystemConfiguration& config = SystemConfiguration::getUnsignedSystemConfig();
    config.setValue(key, to_string(value), ptype);
}

static inline void setUINT64Config(const string& key, const uint64_t& value, const paramType& ptype)
{
    Uint64SystemConfiguration& config = SystemConfiguration::getUint64SystemConfig();
    config.setValue(key, to_string(value), ptype);
}

static inline void setFLOATConfig(const string& key, const float& value, const paramType& ptype)
{
    FloatSystemConfiguration& config = SystemConfiguration::getFloatSystemConfig();
    config.setValue(key, to_string(value), ptype);
}

static inline void setBOOLConfig(const string& key, const bool& value, const paramType& ptype)
{
    BoolSystemConfiguration& config = SystemConfiguration::getBoolSystemConfig();
    config.setValue(key, to_string(value), ptype);
}

class PIMConfiguration
{
   public:
    static RowBufferPolicy getRowBufferPolicy()
    {
        string param = getConfigParam(STRING, "ROW_BUFFER_POLICY");
        if (param == "open_page") {
            return OpenPage;
        } else if (param == "close_page") {
            return ClosePage;
        }
        throw invalid_argument("Invalid row buffer policy");
    }

    static SchedulingPolicy getSchedulingPolicy()
    {
        string param = getConfigParam(STRING, "SCHEDULING_POLICY");

        if (param == "rank_then_bank_round_robin") {
            return RankThenBankRoundRobin;
        } else if (param == "bank_then_rank_round_robin") {
            return BankThenRankRoundRobin;
        }
        throw invalid_argument("Invalid scheduling policy");
    }

    static AddressMappingScheme getAddressMappingScheme()
    {
        string param = getConfigParam(STRING, "ADDRESS_MAPPING_SCHEME");
        for (unsigned i = Scheme1; i <= Scheme7; ++i) {
            string s{"scheme" + to_string(i)};
            if (param == s) {
                return AddressMappingScheme(i);
            }
        }

        /* FIXME: need to change at launch shcha */
        //        if (param == "EagleScheme")
        //        {
        //            return EagleScheme;
        //        }
        //        else if (param == "GPUScheme")
        //        {
        //            return GPUScheme;
        //        }
        //        else if (param == "VegaScheme")
        //        {
        //            return VegaScheme;
        //        }
        //        else if (param == "Scheme8")
        //        {
        //            return Scheme8;
        //        }
        if (param == "Scheme8") {
            return Scheme8;
        } else if (param == "VegaScheme") {
            return VegaScheme;
        }
        throw invalid_argument("Invalid address mapping scheme");
    }

    static QueuingStructure getQueueingStructure()
    {
        string param = getConfigParam(STRING, "QUEUING_STRUCTURE");
        if (param == "per_rank_per_bank") {
            return PerRankPerBank;
        } else if (param == "per_rank") {
            return PerRank;
        }
        throw invalid_argument("Invalid queueing structure");
    }

    static PIMMode getPIMMode()
    {
        string param = getConfigParam(STRING, "PIM_MODE");
        if (param == "mac_in_bankgroup") {
            return mac_in_bankgroup;
        } else if (param == "mac_in_bank") {
            return mac_in_bank;
        }
        throw invalid_argument("Invalid PIM mode");
    }

    static PIMPrecision getPIMPrecision()
    {
        string param = getConfigParam(STRING, "PIM_PRECISION");
        if (param == "FP16") {
            return FP16;
        } else if (param == "INT8") {
            return INT8;
        } else if (param == "FP32") {
            return FP32;
        }
        throw invalid_argument("Invalid PIM precision");
    }

    static int getPIMDataLength()
    {
        string param = getConfigParam(STRING, "PIM_PRECISION");
        if (param == "FP16") {
            return 2;
        } else if (param == "INT8") {
            return 1;
        } else if (param == "FP32") {
            return 4;
        }
        throw invalid_argument("Invalid PIM data length");
    }
};

};  // namespace DRAMSim

#endif
