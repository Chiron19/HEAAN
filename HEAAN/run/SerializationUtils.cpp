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

};

}  // namespace heaan

#endif // !SERIALIZATIONUTILS_CPP