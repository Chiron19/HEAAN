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

void print_shape(complex<double>* mvec, long w, long c, long n, std::string dir="") {
    if (dir != "") {
        ofstream file(dir);
        if (!file.is_open()) {
            cout << "Cannot open file " << dir << endl;
            return;
        }
        for (int i = 0; i < c; i++) {
            for (int j = 0; j < w; j++) {
                for (int k = 0; k < w; k++) {
                    file << setw(8) << setprecision(4) << mvec[i * w * w + j * w + k].real() << " ";
                }
                file << endl;
            }
            file << endl;
        }
        file << endl;
        cout << "Saved to " << dir << endl;
        file.close();
    } else {
        for (int i = 0; i < c; i++) {
            for (int j = 0; j < w; j++) {
                for (int k = 0; k < w; k++) {
                    cout << setw(8) << setprecision(4) << mvec[i * w * w + j * w + k].real() << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }
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
        if (SerializationUtils_::checkLeftRotKey(scheme, i, path) == false) scheme.addLeftRotKey(secretKey, i, path);
    }
}

void generateSerialRightRotKeys(std::set<long>& rotKeys, Scheme_& scheme, SecretKey& secretKey, string path="./serkey") {
    for (long i : rotKeys) {
        if (SerializationUtils_::checkRightRotKey(scheme, i, path) == false) scheme.addRightRotKey(secretKey, i, path);
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

    SerializationUtils_::checkSerialDirectory("./serkey");

    Scheme_ scheme(secretKey, ring, true);

    cout << "keymap: ENCRYPTION    : " << scheme.serKeyMap.at(ENCRYPTION) << endl;
    cout << "keymap: MULTIPLICATION: " << scheme.serKeyMap.at(MULTIPLICATION) << endl;
    
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

    // leftRotKeys.insert({8, 64});
    // rightRotKeys.insert({8, 64});

    generateSerialLeftRotKeys(leftRotKeys, scheme, secretKey);
    generateSerialRightRotKeys(rightRotKeys, scheme, secretKey);
    
    cout << "key done" << endl;

    // Parameters //
    // Total levels: logq / logp = 80
    long logq = 1760; ///< Ciphertext modulus (this value should be <= logQ in "scr/Params.h")
    long logp = 20; ///< Scaling Factor (larger logp will give you more accurate value)
    long logn = 14; ///< number of slot is 2^logn (this value should be < logN in "src/Params.h")
    long n = 1 << logn;
    long slots = n;

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

    // Ciphertext* cipher_temp = SerializationUtils_::readCiphertext("./cipher/layerInit.conv1.bin");
    // std::complex<double>* dec;

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

    // long c = 3;
    // long w = 8;
    // complex<double>* mvec = new complex<double>[n]; // Dynamically allocate memory for the array
    // for (int i = 0; i < c; i++)
    //     for (int j = 0; j < w; j++)
    //         for (int k = 0; k < w; k++)
    //             mvec[i * w * w + j * w + k] = (((j * w + k) / 64.0) - 0.5) * 2.0;
    // print_shape(mvec, w, c, n);
    // Ciphertext cipher_msg;
    // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    // std::complex<double>* dec;
    // timeutils.start("conv 3x3 test");
    // heaan::numThreads = 16;
    // double** weights = new double*[c];
    // for (int i = 0; i < c; i++) {
    //     weights[i] = new double[9];
    //     double temp_weights[9] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    //     std::copy(temp_weights, temp_weights + 9, weights[i]);
    //     cout << endl;
    // }
    // Ciphertext cipher_temp;
    // // scheme.cipherConv3x3(cipher_temp, cipher_msg, weights, scheme, w, c);
    // // scheme.cipherChannelSumAndEqual(cipher_temp, scheme, w, c, 1);
    // double*** kernels = new double**[4];
    // for (int i = 0; i < 4; i++)
    // {
    //     kernels[i] = new double*[3];
    //     for (int j = 0; j < 3; j++)
    //     {
    //         kernels[i][j] = new double[9];
    //         double temp_weights[9] = {0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9};
    //         std::copy(temp_weights, temp_weights + 9, kernels[i][j]);
    //     }
    // }
    // scheme.cipherConv2dLayer(cipher_temp, cipher_msg, kernels, scheme, w, 3, 4);
    // SerializationUtils_::writeCiphertext(cipher_temp, "./temp/conv3x3.bin");
    // timeutils.stop("conv 3x3 test");
    // dec = scheme.decrypt(secretKey, cipher_temp);
    // print_shape(dec, w, 4, n);


    // double* img = new double[1024 * 3];
    // int w, h, c;
    // readImage("./luis.png", img, w, h, c);
    // complex<double>* mvec = new complex<double>[n];
    // for (int i = 0; i < c; i++) {
    //     for (int j = 0; j < w; j++) {
    //         for (int k = 0; k < w; k++) {
    //             mvec[i * w * h + j * h + k] = (img[j * c * h + k * c + i] - 0.5) * 2.0; // Normalize the image and Adjust img[w, h, c] -> mvec[c, w, h]
    //         }
    //     }
    // }
    // for (int i = c * w * h; i < n; i++) mvec[i] = 0;
    // delete[] img;
    // print_shape(mvec, 32, 3, n);
    // Ciphertext cipher_msg;
    // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    // timeutils.start("conv fast");
    // heaan::numThreads = 16;
    // Ciphertext cipher_temp;
    // cipherConv2dLayer_wrapper(cipher_temp, cipher_msg, scheme, 32, 3, 16, "../../weights/conv1.txt");
    // SerializationUtils_::writeCiphertext(cipher_temp, "./temp/conv1.bin");
    // timeutils.stop("conv fast");
    // std::complex<double>* dec = scheme.decrypt(secretKey, cipher_temp);
    // print_shape(dec, 32, 16, n);


    // Ciphertext* cipher_res = SerializationUtils_::readCiphertext("./cipher/layerInit.conv1.bin");
    // cipherBatchNormLayer_wrapper(*cipher_res, scheme, 32, 16, "../../weights/bn1.txt");
    // scheme.cipherReLUAndEqual(*cipher_res, scheme);
    // std::complex<double>* dec = scheme.decrypt(secretKey, *cipher_res);
    // SerializationUtils_::writeCiphertext(*cipher_res, "./temp/layerInit.relu1.bin");
    // print_shape(dec, 32, 16, n, "../../data/Layer0.txt");

    Ciphertext* cipher_msg = SerializationUtils_::readCiphertext("./cipher/layerInit.relu1.bin");
    Ciphertext* cipher_res = new Ciphertext();
    heaan::numThreads = 16;
    *cipher_msg = *SerializationUtils_::readCiphertext("./cipher/layer1.1.bin");
    // cipherConv2dLayerFast_wrapper(*cipher_res, *cipher_msg, scheme, 32, 16, 16, "../../weights/layer1.0.conv1.txt");
    // basicBlock(*cipher_res, *cipher_msg, scheme, 32, 16, {"../../weights/layer1.0.conv1.txt", "../../weights/layer1.0.bn1.txt", "../../weights/layer1.0.conv2.txt", "../../weights/layer1.0.bn2.txt"});
    // basicBlock(*cipher_res, *cipher_msg, scheme, 32, 16, {"../../weights/layer1.1.conv1.txt", "../../weights/layer1.1.bn1.txt", "../../weights/layer1.1.conv2.txt", "../../weights/layer1.1.bn2.txt"});
    layer1(*cipher_res, *cipher_msg, scheme, paths);
    std::complex<double>* dec = scheme.decrypt(secretKey, *cipher_res);
    print_shape(dec, 32, 16, n, "../../data/Layer1.txt");

	return 0;
}
