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

#ifndef PARAMETER_READER_H_
#define PARAMETER_READER_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace DRAMSim
{
class ParameterReaderException : public exception
{
   public:
    ParameterReaderException(string message, unsigned number = 0) : _message(message), _number(number)
    {
        string strNumber = number ? ", line: " + to_string(number) : "";
        _message = _message + strNumber;
    }

    const char* what() const throw() { return _message.c_str(); }

   private:
    string _message;
    unsigned _number;
};

class ParameterReader
{
   public:
    ParameterReader() { throw ParameterReaderException("Constructor without filename is not allowed"); }

    ParameterReader(const string& filename, const bool& isSystemParam = false)
        : _filename(filename), _isSystemParam(isSystemParam)
    {
        _fs.open(filename.c_str());
        if (_fs.fail()) {
            throw ParameterReaderException("Failed to open " + _filename);
        }
        _paramList.clear();
        _paramList.reserve(_paramSize);
    }

    ~ParameterReader() { _fs.close(); }

    vector<pair<string, string>>* getParameter()
    {
        _paramList.clear();
        string line;
        for (unsigned lineNumber = 0; !_fs.eof(); ++lineNumber) {
            getline(_fs, line);
            if (_fs.bad()) {
                throw ParameterReaderException("Failed to read " + _filename);
            }
            if (line.size() == 0) {
                continue;
            }
            // remove space and tab
            removeSpace(line);
            removeTab(line);
            // check whether comment or not
            if (isComment(line)) {
                continue;
            }
            if (!isValid(line)) {
                throw ParameterReaderException(_filename + " has invalid parameter", lineNumber);
            }

            string key = line.substr(0, line.find(equalStr));
            string value = line.find(commentStr) == string::npos
                               ? line.substr(line.find(equalStr) + 1)
                               : line.substr(line.find(equalStr) + 1, line.find(commentStr) - line.find(equalStr) - 1);

            if (key.empty() || value.empty()) {
                throw ParameterReaderException("Cannot parse parameter", lineNumber);
            }

            _paramList.emplace_back(make_pair(key, value));
        }
        return &_paramList;
    }

   private:
    string _filename;
    ifstream _fs;
    bool _isSystemParam = false;
    vector<pair<string, string>> _paramList;

    const char commentStr = ';';
    const char spaceStr = ' ';
    const char tabStr = '\t';
    const char equalStr = '=';
    const unsigned _paramSize = 128;

    void removeStr(string& line, const char& str) { line.erase(remove(line.begin(), line.end(), str), line.end()); }

    void removeSpace(string& line) { removeStr(line, spaceStr); }

    void removeTab(string& line) { removeStr(line, tabStr); }

    bool isValid(string& line) { return getNumberofStr(line, equalStr) == 1 ? true : false; }

    bool isComment(const string& line) { return line.find_first_of(commentStr) == 0 ? true : false; }

    unsigned getNumberofStr(string& line, const char& str) { return count(line.begin(), line.end(), str); }
};

};  // namespace DRAMSim

#endif  // PARAMETER_READER_H_
