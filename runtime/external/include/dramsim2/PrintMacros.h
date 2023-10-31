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

#ifndef PRINT_MACROS_H
#define PRINT_MACROS_H

#include <iostream>
#include <string>

extern std::string SIM_TRACE_FILE;

#define ERROR(str) std::cerr << "[ERROR (" << __FILE__ << ":" << __LINE__ << ")]: " << str << std::endl;

// #define L_BLUE "\x1B[94m"
// #define GREEN "\x1B[32m"
// #define RED "\x1B[31m"
// #define BLUE "\x1B[0;34m"
// #define MAGENTA "\x1B[0;35m"
// #define CYAN "\x1B[0;36m"
// #define GRAY "\x1b[90m"
// #define END "\x1B[0m"

using std::ostream;

#ifdef DEBUG_BUILD
#define DEBUG(str) std::cerr << str << std::endl;
#define DEBUGN(str) std::cerr << str;
#else
#define DEBUG(str)
#define DEBUGN(str)
#endif  // end DEBUG_BUILD

#ifdef NO_OUTPUT
#undef DEBUG
#undef DEBUGN
#define DEBUG(str)
#define DEBUGN(str)
#define PRINT(str)
#define PRINTN(str)
#else
#define PRINT(str)                         \
    do {                                   \
        if (SHOW_SIM_OUTPUT) {             \
            if (LOG_OUTPUT)                \
                dramsimLog << str << endl; \
            else                           \
                cout << str << endl;       \
        }                                  \
    } while (0)
#define PRINTC(color, str)                 \
    do {                                   \
        if (SHOW_SIM_OUTPUT) {             \
            if (LOG_OUTPUT)                \
                dramsimLog << str << endl; \
            else                           \
                cout << str << endl;       \
        }                                  \
    } while (0)
#define PRINTCOND(cond, str)               \
    do {                                   \
        if (SHOW_SIM_OUTPUT && cond) {     \
            if (LOG_OUTPUT)                \
                dramsimLog << str << endl; \
            else                           \
                cout << str << endl;       \
        }                                  \
    } while (0)
#define PRINTN(str)                \
    do {                           \
        if (SHOW_SIM_OUTPUT) {     \
            if (LOG_OUTPUT)        \
                dramsimLog << str; \
            else                   \
                cout << str;       \
        }                          \
    } while (0)
#define PRINTNC(cond, str)             \
    do {                               \
        if (SHOW_SIM_OUTPUT && cond) { \
            if (LOG_OUTPUT)            \
                dramsimLog << str;     \
            else                       \
                cout << str;           \
        }                              \
    } while (0)
#endif  // end NO_OUTPUT

#endif /*PRINT_MACROS_H*/
