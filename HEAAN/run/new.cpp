#include "../src/HEAAN.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <algorithm>
#include <set>
#include <string>
#include <string_view>
#include <stdint.h>
#include <iterator>
#include <iomanip>
#include <iostream>
#include <complex>
#include "Scheme.cpp"
#include "SerializationUtils.cpp"
#include "Layer.cpp"
#include "FormatUtils.cpp"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif // !STB_IMAGE_IMPLEMENTATION

using namespace std;
using namespace NTL;
using namespace heaan;

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
    SerializationUtils_::rotKeysRequirement(leftRotKeys, rightRotKeys, params);
    cout << "leftRotKeys : " << leftRotKeys << endl;
    cout << "rightRotKeys: " << rightRotKeys << endl;

    SerializationUtils_::generateSerialLeftRotKeys(leftRotKeys, scheme, secretKey);
    SerializationUtils_::generateSerialRightRotKeys(rightRotKeys, scheme, secretKey);
    
    timeutils.stop("key generation");

    // Parameters //
    // Total levels: logq / logp
    long logq = 6000; ///< Ciphertext modulus (this value should be <= logQ in "scr/Params.h")
    long logp = 30; ///< Scaling Factor (larger logp will give you more accurate value)
    long logn = 14; ///< number of slot is 2^logn (this value should be < logN in "src/Params.h")
    long n = 1 << logn;
    long slots = n;

    double* img = new double[1024 * 3];
    int w, h, c;
    readImage("./luis.png", img, w, h, c);
    complex<double>* mvec = new complex<double>[n];
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < w; k++) {
                mvec[i * w * h + j * h + k] = (img[j * c * h + k * c + i] - 0.5) * 2.0; // Normalize the image and Adjust img[w, h, c] -> mvec[c, w, h]
            }
        }
    }
    for (int i = c * w * h; i < n; i++) mvec[i] = 0;
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
    
    heaan::SerializationUtils_::checkSerialDirectory("./cipher");

    timeutils.start("encrypt");
    Ciphertext cipher_msg;
    scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    print_shape(scheme.decrypt(secretKey, cipher_msg), 32, 1);
    print_shape(scheme.decrypt(secretKey, cipher_msg), 32, 3, "../../data/img.txt");
    timeutils.stop("encrypt");

    heaan::numThreads = 16;
    
    Ciphertext cipher_res;
    Ciphertext cipher_temp;

    timeutils.start("layerInit");
    if (!SerializationUtils_::checkFile("./cipher/layerInit.relu1.bin")) {
        layerInit(cipher_res, cipher_msg, scheme, paths);
    }
    timeutils.stop("layerInit");

    cipher_res = *SerializationUtils_::readCiphertext("./cipher/layerInit.relu1.bin");
    print_shape(scheme.decrypt(secretKey, cipher_res), 32, 16, "../../data/Layer0.txt");

    timeutils.start("layer1");
    if (!SerializationUtils_::checkFile("./cipher/layer1.3.bin")) {
        cipher_temp = *SerializationUtils_::readCiphertext("./cipher/layerInit.relu1.bin");
        print_shape(scheme.decrypt(secretKey, cipher_temp), 32, 1);
        layer1(cipher_res, cipher_temp, scheme, paths);
    }
    timeutils.stop("layer1");

    cipher_res = *SerializationUtils_::readCiphertext("./cipher/layer1.3.bin");
    print_shape(scheme.decrypt(secretKey, cipher_res), 32, 16, "../../data/Layer1.txt");

    heaan::numThreads = 32;

    timeutils.start("layer2");
    if (!SerializationUtils_::checkFile("./cipher/layer2.3.bin")) {
        cipher_temp = *SerializationUtils_::readCiphertext("./cipher/layer1.3.bin");
        print_shape(scheme.decrypt(secretKey, cipher_temp), 32, 1);
        layer2(cipher_res, cipher_temp, scheme, paths);
    }
    timeutils.stop("layer2");

    cipher_res = *SerializationUtils_::readCiphertext("./cipher/layer2.3.bin");
    print_shape(scheme.decrypt(secretKey, cipher_res), 16, 32, "../../data/Layer2.txt");

    heaan::numThreads = 32;

    timeutils.start("layer3");
    if (!SerializationUtils_::checkFile("./cipher/layer3.3.bin")) {
        cipher_temp = *SerializationUtils_::readCiphertext("./cipher/layer2.3.bin");
        print_shape(scheme.decrypt(secretKey, cipher_temp), 16, 1);
        layer3(cipher_res, cipher_temp, scheme, paths);
    }
    timeutils.stop("layer3");

    cipher_res = *SerializationUtils_::readCiphertext("./cipher/layer3.3.bin");
    print_shape(scheme.decrypt(secretKey, cipher_res), 8, 64, "../../data/Layer3.txt");

    timeutils.start("layerEnd");
    if (!heaan::SerializationUtils_::checkFile("./cipher/layerEnd.linear.bin")) {
        cipher_temp = *heaan::SerializationUtils_::readCiphertext("./cipher/layer3.3.bin");
        print_shape(scheme.decrypt(secretKey, cipher_temp), 8, 1);
        layerEnd(cipher_res, cipher_temp, scheme, paths);
    }
    else {
        cipher_res = *SerializationUtils_::readCiphertext("./cipher/layerEnd.linear.bin");
    }
    timeutils.stop("layerEnd");

    cipher_res = *SerializationUtils_::readCiphertext("./cipher/layerEnd.linear.bin");
    print_shape(scheme.decrypt(secretKey, cipher_res), 8, 10, "../../data/LayerEnd.txt");

    timeutils.start("decrypt");
    complex<double>* dec0 = scheme.decrypt(secretKey, cipher_res);
    print_res_classification(dec0);
    timeutils.stop("decrypt");

    totaltimeutils.stop("program");

    delete[] dec0;
    delete[] mvec;
    
    return 0;
}