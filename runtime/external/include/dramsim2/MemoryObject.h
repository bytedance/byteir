#ifndef MEMORYOBJECT_H
#define MEMORYOBJECT_H

#include <stdint.h>
#include <string>

#include "SimulatorObject.h"
#include "Transaction.h"

namespace DRAMSim
{
class MemoryObject : public SimulatorObject
{
   public:
    MemoryObject() {}
    virtual ~MemoryObject() {}
    virtual bool addTransaction(Transaction* trans) = 0;
    virtual bool addTransaction(bool isWrite, uint64_t addr, BurstType* data) = 0;
    virtual bool addTransaction(bool isWrite, uint64_t addr, const std::string& tag, BurstType* data) = 0;
};

}  // namespace DRAMSim

#endif
