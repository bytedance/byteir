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

#ifndef MEMORYCONTROLLER_H
#define MEMORYCONTROLLER_H

#include <map>
#include <vector>

#include "BankState.h"
#include "BusPacket.h"
#include "CSVWriter.h"
#include "CommandQueue.h"
#include "Configuration.h"
#include "Rank.h"
#include "SimulatorObject.h"
#include "SystemConfiguration.h"
#include "Transaction.h"

using namespace std;

namespace DRAMSim
{
class Rank;
class MemorySystem;
class MemoryControllerStats;
class MemoryController : public SimulatorObject
{
   public:
    // functions
    MemoryController(MemorySystem* ms, CSVWriter& csvOut_, ostream& simLog, Configuration& config);
    virtual ~MemoryController();

    bool addTransaction(Transaction* trans);
    void returnReadData(const Transaction* trans);
    void receiveFromBus(BusPacket* bpacket);
    void attachRanks(vector<Rank*>* ranks);
    void update();
    void printDebugOnUpate();
    void printStats(bool finalStats = false);
    void resetStats();
    bool WillAcceptTransaction();
    bool addBarrier();

    // fields
    vector<Transaction*> transactionQueue;

   private:
    ostream& dramsimLog;
    vector<vector<BankState>> bankStates;

    // functions
    void insertHistogram(unsigned latencyValue, unsigned rank, unsigned bank);
    void updateCommandQueue(BusPacket* poppedBusPacket);
    void updateTransactionQueue();
    void updateBankState();
    void updateRefresh();
    void setBankStatesRW(size_t rank, size_t bank, uint64_t nextRead, uint64_t nextWrite);
    void setBankStates(size_t rank, size_t bank, CurrentBankState currentBankState, BusPacketType lastCommand,
                       uint64_t stateChangeCountdown, uint64_t nextAct);

    // fields
    MemorySystem* parentMemorySystem;

    CommandQueue commandQueue;
    BusPacket* poppedBusPacket;
    vector<BusPacket*> writeDataToSend;
    vector<unsigned> writeDataCountdown;
    vector<Transaction*> returnTransaction;
    vector<Transaction*> pendingReadTransactions;
    map<unsigned, unsigned> latencies;  // latencyValue -> latencyCount
    vector<bool> powerDown;
    vector<Rank*>* ranks;

    // output file
    CSVWriter& csvOut;

    // these packets are counting down waiting to be transmitted on the "bus"
    BusPacket *outgoingCmdPacket, *outgoingDataPacket;
    unsigned cmdCyclesLeft, dataCyclesLeft;

    uint64_t totalTransactions, totalRefreshes;
    vector<uint64_t> grandTotalBankAccesses, totalReadsPerBank, totalWritesPerBank;
    vector<uint64_t> totalReadsPerRank, totalWritesPerRank;
    vector<uint64_t> totalActivatesPerBank, totalActivatesPerRank, totalEpochLatency;
    unsigned refreshRank, refreshBank;
    vector<unsigned> refreshCountdown, refreshCountdownBank;
    Configuration& config;
    MemoryControllerStats* memoryContStats;

   public:
    // energy values are per rank -- SST uses these directly, so make these public
    vector<uint64_t> backgroundEnergy, burstEnergy, actpreEnergy, refreshEnergy, aluPIMEnergy, readPIMEnergy;
    double totalBandwidth;

    uint64_t totalReads, totalWrites;
};

class MemoryControllerStats
{
   public:
    MemoryControllerStats(MemorySystem* parent, CSVWriter& csvOut_, ostream& simLog, Configuration& configuration,
                          uint64_t& totalTrans, vector<uint64_t>& grandTotalBankAcc, vector<uint64_t>& totalReadsPerR,
                          vector<uint64_t>& totalWritesPerR, vector<uint64_t>& totalReadsPerB,
                          vector<uint64_t>& totalWritesPerB, vector<uint64_t>& totalActivatesPerR,
                          vector<uint64_t>& totalActivatesPerB, uint64_t& totalRef, vector<uint64_t>& backgroundE,
                          vector<uint64_t>& burstE, vector<uint64_t>& actpreE, vector<uint64_t>& refreshE,
                          vector<uint64_t>& aluPIME, vector<uint64_t>& readPIME, vector<Transaction*>& pendingReadTrans)
        : csvOut(csvOut_),
          dramsimLog(simLog),
          config(configuration),
          totalTransactions(totalTrans),
          grandTotalBankAccesses(grandTotalBankAcc),
          totalReadsPerRank(totalReadsPerR),
          totalWritesPerRank(totalWritesPerR),
          totalReadsPerBank(totalReadsPerB),
          totalWritesPerBank(totalWritesPerB),
          totalActivatesPerRank(totalActivatesPerR),
          totalActivatesPerBank(totalActivatesPerB),
          totalRefreshes(totalRef),
          backgroundEnergy(backgroundE),
          burstEnergy(burstE),
          actpreEnergy(actpreE),
          refreshEnergy(refreshE),
          aluPIMEnergy(aluPIME),
          readPIMEnergy(readPIME),
          pendingReadTransactions(pendingReadTrans)
    {
        parentMemorySystem = parent;
        totalEpochLatency = vector<uint64_t>(config.NUM_RANKS * config.NUM_BANKS, 0);
        resetStats();
    }

    void printStats(bool finalStats, unsigned myChannel, uint64_t currentClockCycle);
    void insertHistogram(unsigned latencyValue, unsigned rank, unsigned bank);
    void resetStats();

   private:
    MemorySystem* parentMemorySystem;
    ostream& dramsimLog;
    Configuration& config;
    CSVWriter& csvOut;

    uint64_t& totalTransactions;
    vector<uint64_t>& grandTotalBankAccesses;
    vector<uint64_t>& totalReadsPerRank;
    vector<uint64_t>& totalWritesPerRank;
    vector<uint64_t>& totalReadsPerBank;
    vector<uint64_t>& totalWritesPerBank;
    vector<uint64_t>& totalActivatesPerRank;
    vector<uint64_t>& totalActivatesPerBank;
    uint64_t& totalRefreshes;
    vector<uint64_t>& backgroundEnergy;
    vector<uint64_t>& burstEnergy;
    vector<uint64_t>& actpreEnergy;
    vector<uint64_t>& refreshEnergy;
    vector<uint64_t>& aluPIMEnergy;
    vector<uint64_t>& readPIMEnergy;
    vector<Transaction*>& pendingReadTransactions;

    uint64_t currentClockCycle;
    double totalBandwidth;
    map<unsigned, unsigned> latencies;
    vector<uint64_t> totalEpochLatency;
};

}  // namespace DRAMSim

#endif

//
