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

#ifndef __MULTI_CHANNEL_MEMORY_SYSTEM_H__H__
#define __MULTI_CHANNEL_MEMORY_SYSTEM_H__H__

#include <string>
#include <vector>

#include "AddressMapping.h"
#include "CSVWriter.h"
#include "ClockDomain.h"
#include "Configuration.h"
#include "MemoryObject.h"
#include "MemorySystem.h"
#include "SimulatorObject.h"
#include "SystemConfiguration.h"
#include "Transaction.h"

namespace DRAMSim
{
class MultiChannelMemorySystem : public MemoryObject
{
   public:
    MultiChannelMemorySystem(const string& dev, const string& sys, const string& pwd, const string& trc,
                             unsigned megsOfMemory, string* visFilename = NULL);
    virtual ~MultiChannelMemorySystem();

    virtual bool addTransaction(Transaction* trans);
    virtual bool addTransaction(bool isWrite, uint64_t addr, BurstType* data);
    virtual bool addTransaction(bool isWrite, uint64_t addr, const std::string& tag, BurstType* data);

    bool addBarrier(int chanId);

    void update();
    void printStats(bool finalStats = false);
    ostream& getLogFile();
    void RegisterCallbacks(TransactionCompleteCB* readDone, TransactionCompleteCB* writeDone,
                           void (*reportPower)(double bgpower, double burstpower, double refreshpower,
                                               double actprepower));
    unsigned getNumFence(int ch) { return numFence[ch]; }

    void InitOutputFiles(string tracefilename);
    void setCPUClockSpeed(uint64_t cpuClkFreqHz);

    int hasPendingTransactions();

    bool willAcceptTransaction(uint64_t addr);
    bool willAcceptTransaction();

    // output file
    std::ofstream visDataOut;
    ofstream dramsimLog;
    vector<MemorySystem*> channels;
    AddrMapping* addrMapping;

    void getIniBool(const std::string& field, bool* val) { *val = getConfigParam(BOOL, field); }

    void getIniUint(const std::string& field, unsigned int* val) { *val = getConfigParam(UINT, field); }

    void getIniUint64(const std::string& field, uint64_t* val) { *val = getConfigParam(UINT64, field); }

    void getIniFloat(const std::string& field, float* val) { *val = getConfigParam(FLOAT, field); }

   private:
    unsigned findChannelNumber(uint64_t addr);
    uint64_t changeRA12RA13(uint64_t addr);
    void actual_update();

    unsigned megsOfMemory;
    string deviceIniFilename;
    string systemIniFilename;
    string traceFilename;
    string pwd;
    string* visFilename;
    ClockDomain::ClockDomainCrosser clockDomainCrosser;
    static void mkdirIfNotExist(string path);
    static bool fileExists(string path);
    CSVWriter* csvOut;

    double backgroundPower;
    unsigned* numFence;

    Configuration* configuration;
};
}  // namespace DRAMSim

#endif
