#pragma once
#ifndef SERIALIZATIONUTILS_H
#define SERIALIZATIONUTILS_H

#include "../src/HEAAN.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <algorithm>
#include <set>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>

#include "Scheme.h"

using namespace std;
using namespace NTL;

namespace heaan {

class SerializationUtils_ : public SerializationUtils { 
public:

static void readSecretKey(SecretKey &secretKey, string path);
static void checkSerialDirectory(string dir);
static bool checkFile(string path);
static bool deleteFile(string path);
static bool checkLeftRotKey(Scheme &scheme, long r, string dir);
static bool checkRightRotKey(Scheme &scheme, long r, string dir);
static void rotKeysRequirement(std::set<long>& leftRotKeys, std::set<long>& rightRotKeys, std::vector<std::array<long, 4>>& params);
static void generateSerialLeftRotKeys(std::set<long>& rotKeys, Scheme_& scheme, SecretKey& secretKey, string path);
static void generateSerialRightRotKeys(std::set<long>& rotKeys, Scheme_& scheme, SecretKey& secretKey, string path);

};

}  // namespace heaan

#endif // !SERIALIZATIONUTILS_H