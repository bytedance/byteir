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

#ifndef MEMORYSYSTEM_H
#define MEMORYSYSTEM_H

#include <deque>
#include <string>
#include <vector>

#include "AddressMapping.h"
#include "Burst.h"
#include "CSVWriter.h"
#include "Callback.h"
#include "Configuration.h"
#include "MemoryController.h"
#include "MemoryObject.h"
#include "Rank.h"
#include "SimulatorObject.h"
#include "SystemConfiguration.h"
#include "Transaction.h"
#include "Utils.h"

namespace DRAMSim
{
typedef CallbackBase<void, unsigned, uint64_t, uint64_t> Callback_t;

class MemorySystem : public MemoryObject
{
   public:
    // functions
    MemorySystem(unsigned id, unsigned megsOfMemory, CSVWriter& csvOut_, ostream& simLog, Configuration& config);
    virtual ~MemorySystem();
    void update();

    virtual bool addTransaction(Transaction* trans);
    virtual bool addTransaction(bool isWrite, uint64_t addr, BurstType* data);
    virtual bool addTransaction(bool isWrite, uint64_t addr, const std::string& tag, BurstType* data);

    bool addBarrier();
    bool WillAcceptTransaction();
    bool WillAcceptTransaction(uint64_t addr);

    void printStats(bool finalStats);

    void RegisterCallbacks(Callback_t* readDone, Callback_t* writeDone,
                           void (*reportPower)(double bgpower, double burstpower, double refreshpower,
                                               double actprepower));
    // fields
    MemoryController* memoryController;
    vector<Rank*>* ranks;
    deque<Transaction*> pendingTransactions;

    // function pointers
    Callback_t* ReturnReadData;
    Callback_t* WriteDataDone;

    // TODO: make this a functor as well?
    static powerCallBack_t ReportPower;
    unsigned systemID;
    uint64_t numOnTheFlyTransactions;

   private:
    CSVWriter& csvOut;
    ostream& dramsimLog;

    // system and timing parameters
    unsigned num_ranks_;
    Configuration& config;
};
}  // namespace DRAMSim

#endif
