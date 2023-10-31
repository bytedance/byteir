/***************************************************************************************************
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is a property of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system, or translated into any human
 * or computer language in any form by any means,electronic, mechanical, manual or otherwise,
 * or disclosed to third parties without the express written permission of Samsung Electronics.
 * (Use of the Software is restricted to non-commercial, personal or academic, research purpose
 * only)
 **************************************************************************************************/

#ifndef CONFIGURATION_DB_H_
#define CONFIGURATION_DB_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ConfigurationData.h"
#include "ParameterReader.h"

using namespace std;

namespace DRAMSim
{
class ConfigurationDB
{
   public:
    static ConfigurationDB& getDB()
    {
        static ConfigurationDB unique_instance;
        return unique_instance;
    }

    void clearDB(void) { _dbMap.clear(); }
    void initialize(const ConfigurationData* config = defaultConfiguration)
    {
        clearDB();
        if (config != nullptr) {
            for (unsigned i = 0; !config[i].name.empty(); ++i) {
                update(config[i]);
            }
        }
    };

    const ConfigurationData* find(const string& key)
    {
        auto found = _dbMap.find(key);
        return found != _dbMap.end() ? &found->second : nullptr;
    }

    void update(const ConfigurationData& config)
    {
        if (_dbMap.count(config.name) > 0) {
            _dbMap[config.name] = config;
        } else {
            _dbMap.insert(make_pair(config.name, config));
        }
    }

    void update(const vector<pair<string, string>>* configParamList)
    {
        if (!configParamList) {
            return;
        }
        for (auto iter = configParamList->begin(); iter != configParamList->end(); ++iter) {
            if (_dbMap.count(iter->first)) {
                _dbMap[iter->first].value = iter->second;
            }
        }
    }

    void updatefromFile(const string& filename)
    {
        ParameterReader pReader(filename);
        update(pReader.getParameter());
    }

    void dump(std::ofstream& visDataOut)
    {
        visDataOut << "!!SYSTEM INI PARAMETER" << endl;
        for (auto& it : _dbMap) {
            if (it.second.parameterType == SYS_PARAM) {
                visDataOut << it.second.value << endl;
            }
        }
        visDataOut << "!!DEVICE INI PARAMETER" << endl;
        for (auto& it : _dbMap) {
            if (it.second.parameterType == DEV_PARAM) {
                visDataOut << it.second.value << endl;
            }
        }
        visDataOut << "!!EPOCH_DATA" << endl;
    }

   private:
    unordered_map<string, ConfigurationData> _dbMap;
};
};  // namespace DRAMSim

#endif
