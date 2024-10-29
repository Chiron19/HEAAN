#include "../src/HEAAN.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <algorithm>
#include <set>
#include <string>
#include <string_view>
#include <iterator>
#include <iomanip>
#include <iostream>
#include "Scheme.cpp"
#include "SerializationUtils.cpp"
#include "Layer.cpp"

#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using namespace std;
using namespace NTL;
using namespace heaan;

void print_rep_img(double* dec0, long w) {
    for (int i = 0; i < w * 4; i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i] * 1000)) / 1000;
        if ((i + 1) % w == 0) cout << endl;
    }
    cout << " ... " << endl;
    for (int i = w * w; i < w * (w + 4); i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i] * 1000)) / 1000;
        if ((i + 1) % w == 0) cout << endl;
    }
    cout << " ... " << endl;
    for (int i = 2 * w * w; i < 2 * w * (w + 2); i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i] * 1000)) / 1000;
        if ((i + 1) % w == 0) cout << endl;
    }
    cout << endl;
}

void print_rep(complex<double>* dec0, long n) {
    for (int i = 0; i < 10; i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i].real() * 1000)) / 1000 << "," << setw(6) << double(int(dec0[i].imag() * 1000)) / 1000 << "  ";
        if ((i + 1) % 5 == 0) cout << endl;
    }
    cout << " ... " << endl;
    for (int i = n/2 - 5; i < n/2 + 5; i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i].real() * 1000)) / 1000 << "," << setw(6) << double(int(dec0[i].imag() * 1000)) / 1000 << "  ";
        if ((i + 1) % 5 == 0) cout << endl;
    }
    cout << " ... " << endl;
    for (int i = n - 10; i < n; i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i].real() * 1000)) / 1000 << "," << setw(6) << double(int(dec0[i].imag() * 1000)) / 1000 << "  ";
        if ((i + 1) % 5 == 0) cout << endl;
    }
    cout << endl;
}

void print_shape(complex<double>* mvec, long w, long c, long n) {
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < w; k++) {
                cout << setw(6) << mvec[i * w * w + j * w + k].real();
            }
            cout << endl;
        }
        cout << endl;
        cout << endl;
    }
    cout << endl;
}

void print_res_classification(complex<double>* mvec) {
    double max_val = -100;
    int max_idx = -1;
    for (int i = 0; i < 10; i++) {
        cout << setw(8) << mvec[i * 64].real() << " ";
        if (mvec[i * 64].real() > max_val) {
            max_val = mvec[i * 64].real();
            max_idx = i;
        }
    }
    cout << endl;
    cout << "Max value: " << max_val << " at index: " << max_idx << endl;
}

void readImage(string path, double*& image, int& w, int& h, int& c) {
    uint8_t* rgb_image = stbi_load(path.c_str(), &w, &h, &c, 3);
    for (int i = 0; i < w * h * c; i++) {
        image[i] = static_cast<double>(rgb_image[i]) / 255.0;
    }
    stbi_image_free(rgb_image);
}

void generateSerialLeftRotKeys(std::set<long>& rotKeys, Scheme_& scheme, SecretKey& secretKey, string path="./serkey") {
    for (long i : rotKeys) {
        if (SerializationUtils_::checkLeftRotKey(scheme, i, path) == false) scheme.addLeftRotKey(secretKey, i);
    }
}

void generateSerialRightRotKeys(std::set<long>& rotKeys, Scheme_& scheme, SecretKey& secretKey, string path="./serkey") {
    for (long i : rotKeys) {
        if (SerializationUtils_::checkRightRotKey(scheme, i, path) == false) scheme.addRightRotKey(secretKey, i);
    }
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const std::set<T>& set)
{
    if (set.empty())
        return out << "{}";
    out << "{ " << *set.begin();
    std::for_each(std::next(set.begin()), set.end(), [&out](const T& element)
    {
        out << ", " << element;
    });
    return out << " }";
}

int main(int argc, char **argv) {
    cout << "start" << endl;
    srand(time(NULL));
    SetNumThreads(16);
    
    TimeUtils timeutils, totaltimeutils;
    Ring ring;
    SecretKey secretKey(ring);
    SerializationUtils_::readSecretKey(secretKey, "secretKey.bin"); // Read secret key from file path
    Scheme_ scheme(secretKey, ring, true);
    SerializationUtils_::checkSerialDirectory("./serkey"); 

    totaltimeutils.start("program");

    // Key Generation
    timeutils.start("key generation");

    cout << "keymap: ENCRYPTION    : " << scheme.serKeyMap.at(ENCRYPTION) << endl;
    cout << "keymap: MULTIPLICATION: " << scheme.serKeyMap.at(MULTIPLICATION) << endl;
    
    Key* key = SerializationUtils::readKey(scheme.serKeyMap.at(ENCRYPTION));
    cout << "key: " << key->rax[0] << ", " << key->rbx[0] << endl;
    
    std::vector<std::array<long, 4>> params = {
        {{CONV2D, 32, 3, 16}},
        {{CONV2DFAST, 32, 16, 16}},
        {{CONV2DFASTDOWNSAMPLE, 32, 16, 32}},
        {{CONV2DFAST, 16, 32, 32}},
        {{CONV2DFASTDOWNSAMPLE, 16, 32, 64}},
        {{CONV2DFAST, 8, 64, 64}},
        {{AVGPOOL, 8, 64, 64}},
        {{LINEAR, 8, 64, 10}}
    };
    std::set<long> leftRotKeys;
    std::set<long> rightRotKeys;
    rotKeysRequirement(leftRotKeys, rightRotKeys, params);
    cout << "leftRotKeys : " << leftRotKeys << endl;
    cout << "rightRotKeys: " << rightRotKeys << endl;

    generateSerialLeftRotKeys(leftRotKeys, scheme, secretKey);
    generateSerialRightRotKeys(rightRotKeys, scheme, secretKey);
    
    timeutils.stop("key generation");

    // Parameters //
    // Total levels: logq / logp
    long logq = 2000; ///< Ciphertext modulus (this value should be <= logQ in "scr/Params.h")
    long logp = 20; ///< Scaling Factor (larger logp will give you more accurate value)
    long logn = 14; ///< number of slot is 2^logn (this value should be < logN in "src/Params.h")
    long n = 1 << logn;
    long slots = n;

    double* img = new double[1024 * 3];
    int w, h, c;
    readImage("./luis.png", img, w, h, c);
    print_rep_img(img, static_cast<long>(w));

    complex<double>* mvec1 = new complex<double>[n];
    for (int i = 0; i < n; i++) {
        if (i < 1024 * 3) {
            mvec1[i] = img[i];
        } else {
            mvec1[i] = 0;
        }
    }

    delete[] img;

    std::vector<string> paths = {
        "../../weights/conv1.txt",
        "../../weights/bn1.txt",
        "../../weights/layer1.0.conv1.txt",
        "../../weights/layer1.0.bn1.txt",
        "../../weights/layer1.0.conv2.txt",
        "../../weights/layer1.0.bn2.txt",
        "../../weights/layer1.1.conv1.txt",
        "../../weights/layer1.1.bn1.txt",
        "../../weights/layer1.1.conv2.txt",
        "../../weights/layer1.1.bn2.txt",
        "../../weights/layer1.2.conv1.txt",
        "../../weights/layer1.2.bn1.txt",
        "../../weights/layer1.2.conv2.txt",
        "../../weights/layer1.2.bn2.txt",
        "../../weights/layer2.0.conv1.txt",
        "../../weights/layer2.0.bn1.txt",
        "../../weights/layer2.0.conv2.txt",
        "../../weights/layer2.0.bn2.txt",
        "../../weights/layer2.0.downsample.0.txt",
        "../../weights/layer2.0.downsample.1.txt",
        "../../weights/layer2.1.conv1.txt",
        "../../weights/layer2.1.bn1.txt",
        "../../weights/layer2.1.conv2.txt",
        "../../weights/layer2.1.bn2.txt",
        "../../weights/layer2.2.conv1.txt",
        "../../weights/layer2.2.bn1.txt",
        "../../weights/layer2.2.conv2.txt",
        "../../weights/layer2.2.bn2.txt",
        "../../weights/layer3.0.conv1.txt",
        "../../weights/layer3.0.bn1.txt",
        "../../weights/layer3.0.conv2.txt",
        "../../weights/layer3.0.bn2.txt",
        "../../weights/layer3.0.downsample.0.txt",
        "../../weights/layer3.0.downsample.1.txt",
        "../../weights/layer3.1.conv1.txt",
        "../../weights/layer3.1.bn1.txt",
        "../../weights/layer3.1.conv2.txt",
        "../../weights/layer3.1.bn2.txt",
        "../../weights/layer3.2.conv1.txt",
        "../../weights/layer3.2.bn1.txt",
        "../../weights/layer3.2.conv2.txt",
        "../../weights/layer3.2.bn2.txt",
        "../../weights/layer3.3.conv1.txt",
        "../../weights/layer3.3.bn1.txt",
        "../../weights/layer3.3.conv2.txt",
        "../../weights/layer3.3.bn2.txt",
        "../../weights/fc.weight.txt",
        "../../weights/fc.bias.txt",
    };
    
    timeutils.start("encrypt");
    Ciphertext cipher_msg;
    scheme.encrypt(cipher_msg, mvec1, n, logp, logq);
    print_rep(scheme.decrypt(secretKey, cipher_msg), cipher_msg.n);
    timeutils.stop("encrypt");

    heaan::numThreads = 16;
    
    Ciphertext cipher_res;
    Ciphertext cipher_temp;

    timeutils.start("layerInit");
    if (!SerializationUtils_::checkFile("./cipher/layerInit.relu1.bin")) {
        layerInit(cipher_res, cipher_msg, scheme, paths);
    }
    timeutils.stop("layerInit");

    timeutils.start("layer1");
    if (!SerializationUtils_::checkFile("./cipher/layer1.3.bin")) {
        cipher_temp = *SerializationUtils_::readCiphertext("./cipher/layerInit.relu1.bin");
        print_rep(scheme.decrypt(secretKey, cipher_temp), cipher_temp.n);
        layer1(cipher_res, cipher_temp, scheme, paths);
    }
    timeutils.stop("layer1");

    heaan::numThreads = 32;

    timeutils.start("layer2");
    if (!SerializationUtils_::checkFile("./cipher/layer2.3.bin")) {
        cipher_temp = *SerializationUtils_::readCiphertext("./cipher/layer1.3.bin");
        print_rep(scheme.decrypt(secretKey, cipher_temp), cipher_temp.n);
        layer2(cipher_res, cipher_temp, scheme, paths);
    }
    timeutils.stop("layer2");

    heaan::numThreads = 64;

    timeutils.start("layer3");
    if (!SerializationUtils_::checkFile("./cipher/layer3.3.bin")) {
        cipher_temp = *SerializationUtils_::readCiphertext("./cipher/layer2.3.bin");
        print_rep(scheme.decrypt(secretKey, cipher_temp), cipher_temp.n);
        layer3(cipher_res, cipher_temp, scheme, paths);
    }
    timeutils.stop("layer3");

    timeutils.start("layerEnd");
    if (!SerializationUtils_::checkFile("./cipher/layerEnd.bin")) {
        cipher_temp = *SerializationUtils_::readCiphertext("./cipher/layer3.3.bin");
        print_rep(scheme.decrypt(secretKey, cipher_temp), cipher_temp.n);
        layerEnd(cipher_res, cipher_temp, scheme, paths);
    }
    timeutils.stop("layerEnd");

    timeutils.start("decrypt");
    complex<double>* dec0 = scheme.decrypt(secretKey, cipher_res);
    print_res_classification(dec0);
    timeutils.stop("decrypt");

    totaltimeutils.stop("program");

    delete[] dec0;
    delete[] mvec1;

	return 0;
}
