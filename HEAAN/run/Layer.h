#pragma once
#ifndef LAYER_H
#define LAYER_H

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

#include "Scheme.h"
#include "SerializationUtils.h"
#include "FormatUtils.h"

using namespace std;
using namespace NTL;

namespace heaan {

double*** readkernels(string path, long c_out, long c_in);
double** readkernels1x1(string path, long c_out, long c_in);
double** readGammaBeta(string path, long c);
double** readWeights(string path, long c_in, long c_out);
double* readBias(string path, long c);
void freekernels(double*** kernels, long c_out, long c_in);
void freekernels1x1(double** kernels, long c_out, long c_in);
void freeGammaBeta(double** params, long c);
void freeWeights(double** weights, long c_in, long c_out);
void freeBias(double* bias, long c);
void cipherConv2dLayer_wrapper(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c_in, long c_out, string path);
void cipherConv2dLayerFast_wrapper(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c_in, long c_out, string path);
void cipherConv2dLayerFastDownsampling_wrapper(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c, string path);
void cipherConv2d1x1LayerFastDownsampling_wrapper(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c, string path);
void cipherBatchNormLayer_wrapper(Ciphertext& cipher_res, Scheme_& scheme, long w, long c, string path);
void cipherReLUAndEqual_wrapper(Ciphertext& cipher_res, Scheme_& scheme);
void cipherLinearLayer_wrapper(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c_in, long c_out, string path_weight, string path_bias);
void basicBlock(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c, std::vector<string> paths);
void downsamplingBlock(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c, std::vector<string> paths);
void layerInit(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths, SecretKey& secretKey);
void layer1(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths);
void layer2(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths);
void layer3(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths);
void layerEnd(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths);

} // namespace heaan

#endif // !LAYER_H