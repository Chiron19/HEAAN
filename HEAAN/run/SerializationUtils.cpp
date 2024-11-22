#ifndef SERIALIZATIONUTILS_CPP
#define SERIALIZATIONUTILS_CPP

#include "../src/HEAAN.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <string>
#include <iomanip>

using namespace std;
using namespace NTL;

namespace heaan {

static const long CONV2D = 0;
static const long CONV2DFAST = 1;
static const long CONV2DFASTDOWNSAMPLE = 2;
static const long BN = 3;
static const long RELU = 4;
static const long ADD = 5;
static const long DOWNSAMPLE = 6;
static const long DOWNSAMPLEFAST = 7;
static const long AVGPOOL = 8;
static const long LINEAR = 9;

class SerializationUtils_ : public SerializationUtils { 
public:

static void readSecretKey(SecretKey &secretKey, string path="secretKey.bin") {
    unsigned char* secretKeyArray =  new unsigned char[N];

    ifstream infile(path);
    if (infile.good()) {
        infile.close();
        fstream fin;
		fin.open(path, ios::binary|ios::in);
		fin.read(reinterpret_cast<char*>(secretKeyArray), N * sizeof(unsigned char));
		fin.close();
		for (long i = 0; i < N; ++i) {
			secretKey.sx[i] = ZZFromBytes(&secretKeyArray[i], 1);
		}
        cout << path << " found, loaded" << endl;
    } else {
		for (long i = 0; i < N; ++i) {
			BytesFromZZ(&secretKeyArray[i], secretKey.sx[i], 1);
		}
		fstream fout;
		fout.open(path, ios::binary|ios::out);
		fout.write(reinterpret_cast<const char*>(secretKeyArray), N * sizeof(unsigned char));
		fout.close();
		cout << path << " not found, created new one" << endl;
    }
    delete[] secretKeyArray;
}

static void checkSerialDirectory(string dir="./serkey") {
    struct stat info;

    if (stat(dir.c_str(), &info) != 0) {
        cout << "Directory " <<  dir << " does not exist. Creating it." << endl;
        if (mkdir(dir.c_str(), 0777) == -1) {
            cerr << "Error :  " << strerror(errno) << endl;
            return;
        }
    } else if (info.st_mode & S_IFDIR) {
        cout << "Directory " << dir << " exists." << endl;
    } else {
        cerr << dir << " exists but is not a directory." << endl;
        return;
    }
}

static bool checkFile(string path) {
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        cout << "File " << path << " found" << endl;
        return true;
    }
    cout << "File " << path << " not found" << endl;
    return false;
}

static bool deleteFile(string path) {
    if (remove(path.c_str()) != 0) {
        cout << "Error deleting file: " << path << endl;
        return false;
    }
    cout << "File " << path << " deleted" << endl;
    return true;
}

static bool checkLeftRotKey(Scheme &scheme, long r, string dir="./serkey") {
    string path = dir + "/ROTATION_" + to_string(r) + ".txt";
    ifstream rotKeyFile(path);
    if (!rotKeyFile.good()) {
        cout << "Rotation key file not found. Check directory or Call `scheme.addLeftRotKey(" << r << ")` first." << endl;
        return false;
    } else {
        scheme.serLeftRotKeyMap[r] = path;
        cout << "Rotation key file found and loaded to keymap: " << path << endl;
        return true;
    }
}

static bool checkRightRotKey(Scheme &scheme, long r, string dir="./serkey") {
    r = Nh - r;
    string path = dir + "/ROTATION_" + to_string(r) + ".txt";
    ifstream rotKeyFile(path);
    if (!rotKeyFile.good()) {
        cout << "Rotation key file not found. Check directory or Call `scheme.addRightRotKey(" << Nh - r << ")` first." << endl;
        return false;
    } else {
        scheme.serLeftRotKeyMap[r] = path;
        cout << "Rotation key file found and loaded to keymap: " << path << endl;
        return true;
    }
}

/**
 * @brief Determine the requirement of left and right rotation keys for each layer
 * 
 * @param leftRotKeys 
 * @param rightRotKeys 
 * @param params each param = (long[4]){type, w, c_in, c_out}
 */
static void rotKeysRequirement(std::set<long>& leftRotKeys, std::set<long>& rightRotKeys, std::vector<std::array<long, 4>>& params) {
    for (auto& param : params) {
        long type = param[0];
        long w = param[1];
        long c_in = param[2];
        long c_out = param[3];
        if (type == CONV2D) {
            leftRotKeys.insert({1, w, w * w});
            rightRotKeys.insert({1, w, w * w});
            for (long i = 1; i < c_out; i++) rightRotKeys.insert(i * w * w);
        }
        else if (type == CONV2DFAST) {
            // {1, 2, 4, ..., c / 2} * w * w, {1, ..., c_out - 1} * w * w
            leftRotKeys.insert({1, w});
            for (long i = 1; i <= c_in / 2; i <<= 1) leftRotKeys.insert(i * w * w);
            rightRotKeys.insert({1, w});
            for (long i = 1; i < c_out; i++) rightRotKeys.insert(i * w * w);
        }
        else if (type == CONV2DFASTDOWNSAMPLE) {
            leftRotKeys.insert({1, w, 3 * w / 2, 3 * w * w / 4});
            for (long i = 1; i <= c_in / 2; i <<= 1) leftRotKeys.insert(i * w * w / 4);
            rightRotKeys.insert({1, w, w / 2});
            for (long i = 1; i < c_out; i++) rightRotKeys.insert(i * w * w / 4);
            for (long i = 1; i <= w / 4; i <<= 1) leftRotKeys.insert({i, i * 3 * w / 2});
            for (long i = 1; i <= c_in / 2; i <<= 1) leftRotKeys.insert(i * 3 * w * w / 4);
        }
        else if (type == DOWNSAMPLE) {
            leftRotKeys.insert({1, 3 * w / 2, 3 * w * w / 4});
            rightRotKeys.insert({1, w / 2, w * w / 4});
        }
        else if (type == DOWNSAMPLEFAST) {
            // {1, 2, 4, ..., w / 4} * {1, 3 * w / 2}, {1, 2, 4, ..., c / 2} * 3 * w * w / 4
            for (long i = 1; i <= w / 4; i <<= 1) leftRotKeys.insert({i, i * 3 * w / 2});
            for (long i = 1; i <= c_in / 2; i <<= 1) leftRotKeys.insert(i * 3 * w * w / 4);
        }
        else if (type == AVGPOOL) {
            for (long i = w * w / 2; i > 0; i >>= 1) leftRotKeys.insert(i);
        }
        else if (type == LINEAR) {
            for (long i = 1; i <= c_in / 2; i <<= 1) leftRotKeys.insert(i * w * w);
            for (long i = 1; i <= c_out; i <<= 1) rightRotKeys.insert(i * w * w);
        }
    }
}

static void generateSerialLeftRotKeys(std::set<long>& rotKeys, Scheme_& scheme, SecretKey& secretKey, string path="./serkey") {
    for (long i : rotKeys) {
        if (SerializationUtils_::checkLeftRotKey(scheme, i, path) == false) scheme.addLeftRotKey(secretKey, i);
    }
}

static void generateSerialRightRotKeys(std::set<long>& rotKeys, Scheme_& scheme, SecretKey& secretKey, string path="./serkey") {
    for (long i : rotKeys) {
        if (SerializationUtils_::checkRightRotKey(scheme, i, path) == false) scheme.addRightRotKey(secretKey, i);
    }
}


};

}  // namespace heaan

#endif // !SERIALIZATIONUTILS_CPP