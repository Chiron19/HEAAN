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
    // srand(time(NULL));
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
    SerializationUtils_::rotKeysRequirement(leftRotKeys, rightRotKeys, params);
    cout << "leftRotKeys : " << leftRotKeys << endl;
    cout << "rightRotKeys: " << rightRotKeys << endl;

    // leftRotKeys.insert({8, 64});
    // rightRotKeys.insert({8, 64});

    SerializationUtils_::generateSerialLeftRotKeys(leftRotKeys, scheme, secretKey);
    SerializationUtils_::generateSerialRightRotKeys(rightRotKeys, scheme, secretKey);
    
    cout << "key done" << endl;

    // Parameters //
    // Total levels: logq / logp
    long logq = 320; ///< Ciphertext modulus (this value should be <= logQ in "scr/Params.h")
    long logp = 16; ///< Scaling Factor (larger logp will give you more accurate value)
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
    // dec = scheme.decrypt(secretKey, *cipher_temp);
    // for (int i = 0; i < N; i++)
    //     if (secretKey.sx[i] != 0)  cout << i << endl;
    // cout << dec[0] << endl;
    // print_shape(dec, 32, 1);

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
    // print_shape(mvec, 32, 3);
    // Ciphertext cipher_msg;
    // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    // timeutils.start("conv fast");
    // heaan::numThreads = 16;
    // Ciphertext cipher_temp;
    // cipherConv2dLayer_wrapper(cipher_temp, cipher_msg, scheme, 32, 3, 16, "../../weights/conv1.txt");
    // SerializationUtils_::writeCiphertext(cipher_temp, "./cipher/conv1.bin");
    // timeutils.stop("conv fast");
    // std::complex<double>* dec = scheme.decrypt(secretKey, cipher_temp);
    // print_shape(dec, 32, 1);


    // Ciphertext* cipher_res = SerializationUtils_::readCiphertext("./cipher/layerInit.conv1.bin");
    // cipherBatchNormLayer_wrapper(*cipher_res, scheme, 32, 16, "../../weights/bn1.txt");
    // scheme.cipherReLUAndEqual(*cipher_res, scheme);
    // std::complex<double>* dec = scheme.decrypt(secretKey, *cipher_res);
    // SerializationUtils_::writeCiphertext(*cipher_res, "./temp/layerInit.relu1.bin");
    // print_shape(dec, 32, 16, n, "../../data/Layer0.txt");

    // Ciphertext* cipher_msg = SerializationUtils_::readCiphertext("./cipher/layerInit.relu1.bin");
    // Ciphertext* cipher_res = new Ciphertext();
    // heaan::numThreads = 16;
    // *cipher_res = *SerializationUtils_::readCiphertext("./cipher/layerInit.relu1.bin");
    // scheme.cipherReLUAndEqual(*cipher_res, scheme);
    // cipherConv2dLayerFast_wrapper(*cipher_res, *cipher_msg, scheme, 32, 16, 16, "../../weights/layer1.0.conv1.txt");
    // basicBlock(*cipher_res, *cipher_msg, scheme, 32, 16, {"../../weights/layer1.0.conv1.txt", "../../weights/layer1.0.bn1.txt", "../../weights/layer1.0.conv2.txt", "../../weights/layer1.0.bn2.txt"});
    // basicBlock(*cipher_res, *cipher_msg, scheme, 32, 16, {"../../weights/layer1.1.conv1.txt", "../../weights/layer1.1.bn1.txt", "../../weights/layer1.1.conv2.txt", "../../weights/layer1.1.bn2.txt"});
    // layer1(*cipher_res, *cipher_msg, scheme, paths);
    // downsamplingBlock(*cipher_res, *cipher_msßg, scheme, 16, 32, {"../../weights/layer2.0.conv1.txt", "../../weights/layer2.0.bn1.txt", "../../weights/layer2.0.conv2.txt", "../../weights/layer2.0.bn2.txt", "../../weights/layer2.0.downsample.0.txt", "../../weights/layer2.0.downsample.1.txt"});
    // SerializationUtils_::writeCiphertext(*cipher_res, "./cipher/layer2.0.bin");
    // std::complex<double>* dec = scheme.decrypt(secretKey, *cipher_res);
    // print_shape(dec, 32, 16, n, "../../data/relu.txt");

    heaan::numThreads = 16;
    complex<double>* mvec = new complex<double>[n];
    Ciphertext cipher_msg;
    Ciphertext cipher_temp;
    std::complex<double>* dec;
    int c = 3, w = 32;
    // for (int i = 0; i < c; i++) {
    //     for (int j = 0; j < w; j++) {
    //         for (int k = 0; k < w; k++) {
    //             mvec[i * w * w + j * w + k] = (((j * w + k) / 1024.0) - 0.5) * 2.0;
    //         }
    //     }
    // }
    // for (int i = c * w * w; i < n; i++) mvec[i] = 0;
    // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    // timeutils.start("conv");
    // cipherConv2dLayer_wrapper(cipher_temp, cipher_msg, scheme, 32, 3, 16, "../../weights/conv1.txt");
    // timeutils.stop("conv");
    // SerializationUtils_::writeCiphertext(cipher_temp, "./cipher/layerInit.conv1.bin");
    // dec = scheme.decrypt(secretKey, cipher_temp);
    // print_shape(dec, 32, 1);
    // print_shape(dec, 32, 16, "../../data/conv2d_.txt");

    // c = 16, w = 32;
    // for (int i = 0; i < c; i++) {
    //     for (int j = 0; j < w; j++) {
    //         for (int k = 0; k < w; k++) {
    //             mvec[i * w * w + j * w + k] = (((j * w + k) / 1024.0) - 0.5) * 2.0;
    //         }
    //     }
    // }
    // for (int i = c * w * w; i < n; i++) mvec[i] = 0;
    // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    // timeutils.start("conv fast");
    // SerializationUtils_::checkSerialDirectory("./cipher");
    // cipherConv2dLayerFast_wrapper(cipher_temp, cipher_msg, scheme, 32, 16, 16, "../../weights/layer1.0.conv1.txt");
    // cout << cipher_temp.n << " " << cipher_temp.logp << " " << cipher_temp.logq << endl;
    // cout << cipher_temp.ax[0] << " " << cipher_temp.bx[0] << endl;
    // timeutils.stop("conv fast");
    // dec = scheme.decrypt(secretKey, cipher_temp);
    // print_shape(dec, 32, 16, "../../data/conv2dfast1_.txt");

    // c = 32, w = 16;
    // for (int i = 0; i < c; i++) {
    //     for (int j = 0; j < w; j++) {
    //         for (int k = 0; k < w; k++) {
    //             mvec[i * w * w + j * w + k] = (((j * w + k) / 1024.0) - 0.5) * 2.0;
    //         }
    //     }
    // }
    // for (int i = c * w * w; i < n; i++) mvec[i] = 0;
    // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    // timeutils.start("conv fast");
    // SerializationUtils_::checkSerialDirectory("./cipher");
    // cipherConv2dLayerFast_wrapper(cipher_temp, cipher_msg, scheme, 16, 32, 32, "../../weights/layer2.1.conv1.txt");
    // cout << cipher_temp.n << " " << cipher_temp.logp << " " << cipher_temp.logq << endl;
    // cout << cipher_temp.ax[0] << " " << cipher_temp.bx[0] << endl;
    // timeutils.stop("conv fast");
    // dec = scheme.decrypt(secretKey, cipher_temp);
    // print_shape(dec, 16, 32, "../../data/conv2dfast2_.txt");

    // c = 64, w = 8;
    // for (int i = 0; i < c; i++) {
    //     for (int j = 0; j < w; j++) {
    //         for (int k = 0; k < w; k++) {
    //             mvec[i * w * w + j * w + k] = (((j * w + k) / 1024.0) - 0.5) * 2.0;
    //         }
    //     }
    // }
    // for (int i = c * w * w; i < n; i++) mvec[i] = 0;
    // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    // timeutils.start("conv fast");
    // SerializationUtils_::checkSerialDirectory("./cipher");
    // cipherConv2dLayerFast_wrapper(cipher_temp, cipher_msg, scheme, 8, 64, 64, "../../weights/layer3.1.conv1.txt");
    // cout << cipher_temp.n << " " << cipher_temp.logp << " " << cipher_temp.logq << endl;
    // cout << cipher_temp.ax[0] << " " << cipher_temp.bx[0] << endl;
    // timeutils.stop("conv fast");
    // dec = scheme.decrypt(secretKey, cipher_temp);
    // print_shape(dec, 8, 64, "../../data/conv2dfast3_.txt");

    // c = 16, w = 32;
    // for (int i = 0; i < c; i++) {
    //     for (int j = 0; j < w; j++) {
    //         for (int k = 0; k < w; k++) {
    //             mvec[i * w * w + j * w + k] = (((j * w + k) / 1024.0) - 0.5) * 2.0;
    //         }
    //     }
    // }
    // for (int i = c * w * w; i < n; i++) mvec[i] = 0;
    // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    // timeutils.start("conv fast downsample");
    // cipherConv2dLayerFastDownsampling_wrapper(cipher_temp, cipher_msg, scheme, 16, 32, "../../weights/layer2.0.conv1.txt");
    // timeutils.stop("conv fast downsample");
    // dec = scheme.decrypt(secretKey, cipher_temp);
    // print_shape(dec, 16, 32, "../../data/conv2dfastdownsample1_.txt");

    // c = 32, w = 16;
    // for (int i = 0; i < c; i++) {
    //     for (int j = 0; j < w; j++) {
    //         for (int k = 0; k < w; k++) {
    //             mvec[i * w * w + j * w + k] = (((j * w + k) / 1024.0) - 0.5) * 2.0;
    //         }
    //     }
    // }
    // for (int i = c * w * w; i < n; i++) mvec[i] = 0;
    // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    // timeutils.start("conv fast downsample");
    // cipherConv2dLayerFastDownsampling_wrapper(cipher_temp, cipher_msg, scheme, 8, 64, "../../weights/layer3.0.conv1.txt");
    // timeutils.stop("conv fast downsample");
    // dec = scheme.decrypt(secretKey, cipher_temp);
    // print_shape(dec, 8, 64, "../../data/conv2dfastdownsample2_.txt");

    // c = 16, w = 32;
    // for (int i = 0; i < c; i++) {
    //     for (int j = 0; j < w; j++) {
    //         for (int k = 0; k < w; k++) {
    //             mvec[i * w * w + j * w + k] = (((j * w + k) / 1024.0) - 0.5) * 2.0;
    //         }
    //     }
    // }
    // for (int i = c * w * w; i < n; i++) mvec[i] = 0;
    // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    // timeutils.start("bn");
    // cipherBatchNormLayer_wrapper(cipher_msg, scheme, 32, 16, "../../weights/bn1.txt");
    // timeutils.stop("bn");
    // dec = scheme.decrypt(secretKey, cipher_msg);
    // print_shape(dec, 32, 16, "../../data/bn_.txt");

    // c = 16, w = 32;
    // for (int i = 0; i < c; i++) {
    //     for (int j = 0; j < w; j++) {
    //         for (int k = 0; k < w; k++) {
    //             mvec[i * w * w + j * w + k] = (((j * w + k) / 1024.0) - 0.5) * 2.3;
    //         }
    //     }
    // }
    // for (int i = c * w * w; i < n; i++) mvec[i] = 0;
    // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    // timeutils.start("relu");
    // scheme.cipherReLUAndEqual(cipher_msg, scheme, 1);
    // timeutils.stop("relu");
    // dec = scheme.decrypt(secretKey, cipher_msg);
    // print_shape(dec, 32, 16, "../../data/relu1.txt");

    // c = 16, w = 32;
    // for (int i = 0; i < c; i++) {
    //     for (int j = 0; j < w; j++) {
    //         for (int k = 0; k < w; k++) {
    //             mvec[i * w * w + j * w + k] = (((j * w + k) / 1024.0) - 0.5) * 2.3;
    //         }
    //     }
    // }
    // for (int i = c * w * w; i < n; i++) mvec[i] = 0;
    // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    // timeutils.start("relu");
    // scheme.cipherReLUAndEqual(cipher_msg, scheme, 2);
    // timeutils.stop("relu");
    // dec = scheme.decrypt(secretKey, cipher_msg);
    // print_shape(dec, 32, 16, "../../data/relu2.txt");
    
    // c = 64, w = 8;
    // for (int i = 0; i < c; i++) {
    //     for (int j = 0; j < w; j++) {
    //         for (int k = 0; k < w; k++) {
    //             mvec[i * w * w + j * w + k] = ((i * (j * w + k) / 1024.0) - 0.5) * 2.0;
    //         }
    //     }
    // }
    // for (int i = c * w * w; i < n; i++) mvec[i] = 0;
    // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    // timeutils.start("avg pool");
    // scheme.cipherAvgPoolingAndEqual(cipher_msg, scheme, 8, 64);
    // timeutils.stop("avg pool");
    // dec = scheme.decrypt(secretKey, cipher_msg);
    // print_shape(dec, 8, 64, "../../data/pool_.txt");

    // c = 64, w = 8;
    // for (int i = 0; i < c; i++) {
    //     for (int j = 0; j < w; j++) {
    //         for (int k = 0; k < w; k++) {
    //             mvec[i * w * w + j * w + k] = ((i * (j * w + k) / 1024.0) - 0.5) * 2.0;
    //         }
    //     }
    // }
    // for (int i = c * w * w; i < n; i++) mvec[i] = 0;
    // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
    // print_res_classification(scheme.decrypt(secretKey, cipher_msg));
    // timeutils.start("linear");
    // cipherLinearLayer_wrapper(cipher_temp, cipher_msg, scheme, 8, 64, 10, "../../weights/fc.weight.txt", "../../weights/fc.bias.txt");
    // timeutils.stop("linear");
    // dec = scheme.decrypt(secretKey, cipher_temp);
    // print_res_classification(dec);

    for (int v = 1; v <=10; v++) {
        srand(v);
        c = 64, w = 8;
        string str = "../../data/input_" + to_string(v) + ".txt";
        istream* is = new ifstream(str);
        for (int i = 0; i < c; i++) {
            for (int j = 0; j < w; j++) {
                for (int k = 0; k < w; k++) {
                    *is >> mvec[i * w * w + j * w + k];
                }
            }
        }
        for (int i = c * w * w; i < n; i++) mvec[i] = 0;
        complex<double>* mvec1 = new complex<double>[n];
        memccpy(mvec1, mvec, n, sizeof(complex<double>));
        // string str = "../../data/input_" + to_string(v) + ".txt";
        // print_shape(mvec, 8, 64, str);
        scheme.encrypt(cipher_msg, mvec, n, logp, logq);
        timeutils.start("avg pool");
        scheme.cipherAvgPoolingAndEqual(cipher_msg, scheme, 8, 64);
        timeutils.stop("avg pool");
        dec = scheme.decrypt(secretKey, cipher_msg);
        str = "../../data/pool" + to_string(v) + ".txt";
        print_shape(dec, 8, 64, str);
        // scheme.encrypt(cipher_msg, mvec, n, logp, logq);
        // cipher_temp.free();
        // timeutils.start("linear");
        // cipherLinearLayer_wrapper(cipher_temp, cipher_msg, scheme, 8, 64, 10, "../../weights/fc.weight.txt", "../../weights/fc.bias.txt");
        // timeutils.stop("linear");
        // dec = scheme.decrypt(secretKey, cipher_temp);
        // print_res_classification(dec);
    }
    

	return 0;
}