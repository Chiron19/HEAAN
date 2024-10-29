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

void print_shape(complex<double>* mvec, long w, long c, long n) {
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < w; k++) {
                if (mvec[i * w * w + j * w + k].real() < 0.001) {
                    cout << setw(7) << setprecision(4) << 0.0 << " ";
                } else {
                    cout << setw(7) << int(round(1000.0 * mvec[i * w * w + j * w + k].real()) / 1000) << " ";
                }
            }
            cout << endl;
        }
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
    SetNumThreads(16);
    TimeUtils timeutils;
    Ring ring;
    SecretKey secretKey(ring);
    SerializationUtils_::readSecretKey(secretKey, "secretKey.bin");
    Scheme_ scheme(secretKey, ring, true);
    SerializationUtils_::checkSerialDirectory("./serkey");

    cout << "keymap: ENCRYPTION    : " << scheme.serKeyMap.at(ENCRYPTION) << endl;
    cout << "keymap: MULTIPLICATION: " << scheme.serKeyMap.at(MULTIPLICATION) << endl;
    
    std::vector<std::array<long, 4>> params = {
        {{CONV2DFAST, 32, 16, 16}},
    };
    std::set<long> leftRotKeys;
    std::set<long> rightRotKeys;
    rotKeysRequirement(leftRotKeys, rightRotKeys, params);
    cout << "leftRotKeys : " << leftRotKeys << endl;
    cout << "rightRotKeys: " << rightRotKeys << endl;

    generateSerialLeftRotKeys(leftRotKeys, scheme, secretKey, "./temp");
    generateSerialRightRotKeys(rightRotKeys, scheme, secretKey, "./temp");
    
    cout << "key done" << endl;

    // Parameters //
    // Total levels: logq / logp = 80
    long logq = 1760; ///< Ciphertext modulus (this value should be <= logQ in "scr/Params.h")
    long logp = 20; ///< Scaling Factor (larger logp will give you more accurate value)
    long logn = 14; ///< number of slot is 2^logn (this value should be < logN in "src/Params.h")
    long n = 1 << logn;
    long slots = n;
    long c = 16;
    long w = 32;

    complex<double>* mvec = new complex<double>[n]; // Dynamically allocate memory for the array
    for (int i = 0; i < c; i++) 
    {
        for (int j = 0; j < w; j++)
        {
            for (int k = 0; k < w; k++)
            {
                mvec[i * w * w + j * w + k] = j * w + k;
            }
            
        }
    }
    print_shape(mvec, w, c, n);

    Ciphertext cipher_msg;
    scheme.encrypt(cipher_msg, mvec, n, logp, logq);

    Ciphertext cipher_temp;
    std::complex<double>* dec;

    // timeutils.start("downsample row");
    // scheme.cipherDownsamplingRow(cipher_temp, cipher_msg, scheme, w, c);
    // timeutils.stop("downsample row");

    // dec = scheme.decrypt(secretKey, cipher_temp);
    // print_shape(dec, w, c, n);

    // timeutils.start("downsample fast");
    // scheme.cipherDownsamplingRowFast(cipher_temp, cipher_msg, scheme, w, c);
    // scheme.cipherDownsamplingColumnFast(cipher_msg, cipher_temp, scheme, w, c);
    // scheme.cipherDownsamplingChannelFast(cipher_temp, cipher_msg, scheme, w, c);
    // cipher_temp.n >>= 1;
    // w >>= 1;
    // timeutils.stop("downsample fast");

    timeutils.start("conv fast");
    heaan::numThreads = 1;

    // double** weights = new double*[c];
    // for (int i = 0; i < c; i++) {
    //     weights[i] = new double[9];
    //     for (int j = 0; j < 9; j++) {
    //         weights[i][j] = (rand() % 1000) / 1000.0;
    //         cout << setw(6) << weights[i][j] << " ";
    //     }
    //     cout << endl;
    // }
    
    cipherConv2dLayerFast_wrapper(cipher_temp, cipher_msg, scheme, w, c, c, "../../weights/layer1.0.conv1.txt");
    // scheme.cipherConv3x3(cipher_temp, cipher_msg, weights, scheme, w, c);
    timeutils.stop("conv fast");

    dec = scheme.decrypt(secretKey, cipher_temp);
    print_shape(dec, w, c, n);

	return 0;
}
