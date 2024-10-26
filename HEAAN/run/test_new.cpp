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
    for (int i = 0; i < 100; i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i].real() * 1000)) / 1e9 << "," << setw(6) << double(int(dec0[i].imag() * 1000)) / 1000 << "  ";
        if ((i + 1) % 5 == 0) cout << endl;
    }
    cout << " ... " << endl;
    for (int i = n/2-5; i < n/2 + 5; i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i].real() * 1000)) / 1e9 << "," << setw(6) << double(int(dec0[i].imag() * 1000)) / 1000 << "  ";
        if ((i + 1) % 5 == 0) cout << endl;
    }
    cout << " ... " << endl;
    for (int i = n - 5; i < n; i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i].real() * 1000)) / 1e9 << "," << setw(6) << double(int(dec0[i].imag() * 1000)) / 1000 << "  ";
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
    for (int i = 0; i < 10; i++) {
        cout << setw(8) << mvec[i * 64].real() << endl;
    }
    cout << endl;
}

void print_res_pool(complex<double>* mvec) {
    for (int i = 0; i < 64; i++) {
        cout << setw(8) << mvec[i * 64].real() << endl;
    }
    cout << endl;
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
    rotKeysRequirement(leftRotKeys, rightRotKeys, params);
    cout << "leftRotKeys : " << leftRotKeys << endl;
    cout << "rightRotKeys: " << rightRotKeys << endl;

    generateSerialLeftRotKeys(leftRotKeys, scheme, secretKey);
    generateSerialRightRotKeys(rightRotKeys, scheme, secretKey);
    
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
    // for (int i = 0; i < 3; i++) {
    //     for (int j = 0; j < 1024; j++) {
    //         img[i * 1024 + j] = 0.1; // Assign a random value to each entry
    //     }
    // }


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
        "./weights/conv1.txt",
        "./weights/bn1.txt",
        "./weights/layer1.0.conv1.txt",
        "./weights/layer1.0.bn1.txt",
        "./weights/layer1.0.conv2.txt",
        "./weights/layer1.0.bn2.txt",
        "./weights/layer1.1.conv1.txt",
        "./weights/layer1.1.bn1.txt",
        "./weights/layer1.1.conv2.txt",
        "./weights/layer1.1.bn2.txt",
        "./weights/layer1.2.conv1.txt",
        "./weights/layer1.2.bn1.txt",
        "./weights/layer1.2.conv2.txt",
        "./weights/layer1.2.bn2.txt",
        "./weights/layer2.0.conv1.txt",
        "./weights/layer2.0.bn1.txt",
        "./weights/layer2.0.conv2.txt",
        "./weights/layer2.0.bn2.txt",
        "./weights/layer2.0.downsample.0.txt",
        "./weights/layer2.0.downsample.1.txt",
        "./weights/layer2.1.conv1.txt",
        "./weights/layer2.1.bn1.txt",
        "./weights/layer2.1.conv2.txt",
        "./weights/layer2.1.bn2.txt",
        "./weights/layer2.2.conv1.txt",
        "./weights/layer2.2.bn1.txt",
        "./weights/layer2.2.conv2.txt",
        "./weights/layer2.2.bn2.txt",
        "./weights/layer3.0.conv1.txt",
        "./weights/layer3.0.bn1.txt",
        "./weights/layer3.0.conv2.txt",
        "./weights/layer3.0.bn2.txt",
        "./weights/layer3.0.downsample.0.txt",
        "./weights/layer3.0.downsample.1.txt",
        "./weights/layer3.1.conv1.txt",
        "./weights/layer3.1.bn1.txt",
        "./weights/layer3.1.conv2.txt",
        "./weights/layer3.1.bn2.txt",
        "./weights/layer3.2.conv1.txt",
        "./weights/layer3.2.bn1.txt",
        "./weights/layer3.2.conv2.txt",
        "./weights/layer3.2.bn2.txt",
        "./weights/layer3.3.conv1.txt",
        "./weights/layer3.3.bn1.txt",
        "./weights/layer3.3.conv2.txt",
        "./weights/layer3.3.bn2.txt",
        "./weights/fc.weight.txt",
        "./weights/fc.bias.txt",
    };

    Ciphertext cipher_msg;
    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layerInit.conv1.bin");
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    // print_rep(scheme.decrypt(secretKey, cipher_msg), cipher_msg.n);

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layerInit.bn1.bin");
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    // print_rep(scheme.decrypt(secretKey, cipher_msg), cipher_msg.n);

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layerInit.relu1.bin");
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    // print_rep(scheme.decrypt(secretKey, cipher_msg), cipher_msg.n);

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer1.1.bin");
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    // print_rep(scheme.decrypt(secretKey, cipher_msg), cipher_msg.n);
    
    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer1.2.bin");
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    // print_rep(scheme.decrypt(secretKey, cipher_msg), cipher_msg.n);

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer1.3.bin");
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    // print_rep(scheme.decrypt(secretKey, cipher_msg), cipher_msg.n);

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer2.1.bin");
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    // print_rep(scheme.decrypt(secretKey, cipher_msg), cipher_msg.n);

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer2.2.bin");
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    // print_rep(scheme.decrypt(secretKey, cipher_msg), cipher_msg.n);

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer2.3.bin");
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    // print_rep(scheme.decrypt(secretKey, cipher_msg), cipher_msg.n);

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer3.1.bin");
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    // print_rep(scheme.decrypt(secretKey, cipher_msg), cipher_msg.n);

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer3.2.bin");
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer3.3.bin");
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    // // print_rep(scheme.decrypt(secretKey, cipher_msg), cipher_msg.n);
    // Ciphertext cipher_temp;
    // scheme.cipherAvgPoolingAndEqual(cipher_msg, scheme, 8, 64);
    // cout << "layerEnd.avgpool done" << endl;
    // cout << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    // // print_res_pool(scheme.decrypt(secretKey, cipher_msg));
    
    // cipherLinearLayer_wrapper(cipher_temp, cipher_msg, scheme, 8, 64, 10, "./weights/fc.weight.txt", "./weights/fc.bias.txt");
    // cout << "layerEnd.linear done" << endl;
    // cout << cipher_temp.n << ", " << cipher_temp.logp << ", " << cipher_temp.logq << endl;

    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layerEnd.avgpool.bin");
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    
    cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layerEnd.linear.bin");
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;

    std::complex<double>* dec = scheme.decrypt(secretKey, cipher_msg);
    print_res_classification(dec);
	return 0;
}
