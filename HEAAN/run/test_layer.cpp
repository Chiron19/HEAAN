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
#include "FormatUtils.cpp"

#include <stdint.h>

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
    cout << "start" << endl;
    srand(time(NULL));
    SetNumThreads(8);
    TimeUtils timeutils;
    Ring ring;
    SecretKey secretKey(ring);
    SerializationUtils_::readSecretKey(secretKey, "secretKey.bin");
    Scheme_ scheme(secretKey, ring, true);
    SerializationUtils_::checkSerialDirectory("./serkey");

    cout << "keymap: ENCRYPTION    : " << scheme.serKeyMap.at(ENCRYPTION) << endl;
    cout << "keymap: MULTIPLICATION: " << scheme.serKeyMap.at(MULTIPLICATION) << endl;
    
    Key* key = SerializationUtils::readKey(scheme.serKeyMap.at(ENCRYPTION));
    cout << "key: " << key->rax[0] << ", " << key->rbx[0] << endl;
    // Key Generation
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
    
    cout << "key done" << endl;

    // Parameters //
    // Total levels: logq / logp = 80
    long logq = 800; ///< Ciphertext modulus (this value should be <= logQ in "scr/Params.h")
    long logp = 10; ///< Scaling Factor (larger logp will give you more accurate value)
    long logn = 14; ///< number of slot is 2^logn (this value should be < logN in "src/Params.h")
    long n = 1 << logn;
    long slots = n;

    double* img = new double[1024 * 3]; // Dynamically allocate memory for the array
    int w, h, c;
    readImage("./luis.png", img, w, h, c);
    print_rep_img(img, static_cast<long>(w));


    complex<double>* mvec1 = new complex<double>[n]; // Dynamically allocate memory for the array
    for (int i = 0; i < n; i++) {
        if (i < 1024 * 3) {
            mvec1[i] = img[i];
        } else {
            mvec1[i] = 0;
        }
    }

    delete[] img;

    cout << "img prepare to encrypt done" << endl;

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

    Ciphertext cipher_msg;
    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layerInit.conv1.bin");
    cout << "Layer Init conv: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    print_shape(scheme.decrypt(secretKey, cipher_msg), 32, 16, "../../data/layerInit.conv1.txt");

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layerInit.bn1.bin");
    cout << "Layer Init bn: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    print_shape(scheme.decrypt(secretKey, cipher_msg), 32, 16, "../../data/layerInit.bn1.txt");

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layerInit.relu1.bin");
    cout << "Layer Init relu: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    print_shape(scheme.decrypt(secretKey, cipher_msg), 32, 16, "../../data/layerInit.relu1.txt");
    
    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer1.1.bin");
    cout << "Layer1.1: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    print_shape(scheme.decrypt(secretKey, cipher_msg), 32, 16, "../../data/layer1.1.txt");

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer1.2.bin");
    cout << "Layer1.2: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    print_shape(scheme.decrypt(secretKey, cipher_msg), 32, 16, "../../data/layer1.2.txt");

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer1.3.bin");
    cout << "Layer1.3: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    print_shape(scheme.decrypt(secretKey, cipher_msg), 32, 16, "../../data/layer1.3.txt");

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer2.1.bin");
    cout << "Layer2.1: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    print_shape(scheme.decrypt(secretKey, cipher_msg), 16, 32, "../../data/layer2.1.txt");

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer2.2.bin");
    cout << "Layer2.2: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    print_shape(scheme.decrypt(secretKey, cipher_msg), 16, 32, "../../data/layer2.2.txt");

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer2.3.bin");
    cout << "Layer2.3: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    print_shape(scheme.decrypt(secretKey, cipher_msg), 16, 32, "../../data/layer2.3.txt");

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer3.1.bin");
    cout << "Layer3.1: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    print_shape(scheme.decrypt(secretKey, cipher_msg), 8, 64, "../../data/layer3.1.txt");

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer3.2.bin");
    cout << "Layer3.2: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    print_shape(scheme.decrypt(secretKey, cipher_msg), 8, 64, "../../data/layer3.2.txt");

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer3.3.bin");
    cout << "Layer3.3: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    print_shape(scheme.decrypt(secretKey, cipher_msg), 8, 64, "../../data/layer3.3.txt");

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layerEnd.linear.bin");
    cout << "Layer End: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    std::complex<double>* dec = scheme.decrypt(secretKey, cipher_msg);
    print_res_classification(dec);

	return 0;
}
