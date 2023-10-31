#ifndef __CLOCKDOMAIN__
#define __CLOCKDOMAIN__

#include <stdint.h>
#include <cmath>
#include <iostream>

namespace ClockDomain
{
template <typename ReturnT>
class CallbackBase
{
   public:
    virtual ReturnT operator()() = 0;
    virtual ~CallbackBase() {}
};

template <typename ConsumerT, typename ReturnT>
class Callback : public CallbackBase<ReturnT>
{
   private:
    typedef ReturnT (ConsumerT::*PtrMember)();

   public:
    Callback(ConsumerT* const object, PtrMember member) : object(object), member(member) {}

    Callback(const Callback<ConsumerT, ReturnT>& e) : object(e.object), member(e.member) {}

    virtual ~Callback() {}

    ReturnT operator()() { return (const_cast<ConsumerT*>(object)->*member)(); }

   private:
    ConsumerT* const object;
    const PtrMember member;
};

typedef CallbackBase<void> ClockUpdateCB;

class ClockDomainCrosser
{
   public:
    ClockUpdateCB* callback;
    uint64_t clock1, clock2;
    uint64_t counter1, counter2;
    ClockDomainCrosser(ClockUpdateCB* _callback);
    // ClockDomainCrosser(uint64_t _clock1, uint64_t _clock2, ClockUpdateCB
    // *_callback);
    // ClockDomainCrosser(double ratio, ClockUpdateCB *_callback);
    void update();
    virtual ~ClockDomainCrosser()
    {
        if (callback) {
            delete callback;
        }
    }
};

/*
    class TestObj
    {
            public:
            TestObj() {}
            void cb();
            int test();
    };
*/
}  // namespace ClockDomain
#endif
