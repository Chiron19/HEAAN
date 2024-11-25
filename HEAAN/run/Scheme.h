#pragma once
#ifndef SCHEME_H
#define SCHEME_H

#include "../src/HEAAN.h"
#include <complex>

namespace heaan {

class Scheme_;

static long numThreads = 1;

struct ThreadData {
    Ciphertext* cipher_conv;
    Ciphertext* cipher_msg;
    double*** kernels;
    Scheme_* scheme;
    long w;
    long c_in;
    long kernel_index_start;
    long kernel_index_end;
    long cipher_index_start;
};

class Scheme_: public Scheme {
public:

Scheme_(SecretKey& secretKey, Ring& ring, bool isSerialized = false, std::string dir="./serkey") 
: Scheme(secretKey, ring, isSerialized, dir) {}

void packKernelConst(std::complex<double>* mvec, long n, double* kernel, long w, long c);
void packKernel(std::complex<double>** mvec, double** kernel, long w, long c);
void packConst(std::complex<double>* mvec, long n, double* const_vec, long w, long c, double const_scale=1.0);
void packWeights(std::complex<double>* mvec, long n, double* weights, long w, long c);
void maskSlot(std::complex<double>* mvec, long n, long w, long c_start_id);
void maskSlotRow(std::complex<double>* mvec, long n, long c, long w, long step);
void maskSlotColumn(std::complex<double>* mvec, long n, long c, long w, long step);
void maskSlotChannel(std::complex<double>* mvec, long n, long c, long w, long step);
void cipherConv3x3(Ciphertext& cipher_res, Ciphertext& cipher_msg, double** kernel, Scheme_ &scheme, long w, long c);
void cipherConv1x1(Ciphertext& cipher_res, Ciphertext& cipher_msg, double* kernel, Scheme_ &scheme, long w, long c);
void cipherChannelSumAndEqual(Ciphertext& cipher, Scheme_ &scheme, long w, long c, long c_target_id);
void cipherChannelFastSumAndEqual(Ciphertext& cipher, Scheme_ &scheme, long w, long c, long c_target_id);
void cipherConv2dLayer(Ciphertext &cipher_res, Ciphertext &cipher_msg, double*** kernels, Scheme_ &scheme, long w, long c_in, long c_out);
void cipherConv2dLayerFast(Ciphertext &cipher_res, Ciphertext &cipher_msg, double*** kernels, Scheme_ &scheme, long w, long c_in, long c_out);
void cipherConv2dLayerFastDownsampling(Ciphertext &cipher_res, Ciphertext &cipher_msg, double*** kernels, Scheme_ &scheme, long w, long c_in);
void cipherConv2d1x1LayerFastDownsampling(Ciphertext &cipher_res, Ciphertext &cipher_msg, double** kernels, Scheme_ &scheme, long w, long c_in);
void cipherDownsamplingRow(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_ &scheme, long w, long c);
void cipherDownsamplingColumn(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_ &scheme, long w, long c);
void cipherDownsamplingChannel(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_ &scheme, long w, long c);
void cipherDownsampling(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_ &scheme, long w, long c);
void cipherDownsamplingRowFast(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_ &scheme, long w, long c);
void cipherDownsamplingColumnFast(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_ &scheme, long w, long c);
void cipherDownsamplingChannelFast(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_ &scheme, long w, long c);
void cipherDownsamplingFast(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_ &scheme, long w, long c);
void cipherBatchNormLayer(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_ &scheme, long w, long c, double* gamma, double* beta);
void cipherBatchNormLayerAndEqual(Ciphertext& cipher_msg, Scheme_ &scheme, long w, long c, double* gamma, double* beta, double const_scale=1.0);
void cipherReLUAndEqual(Ciphertext& cipher_msg, Scheme_ &scheme, long opt=0);
void cipherAvgPoolingAndEqual(Ciphertext& cipher_msg, Scheme_ &scheme, long w, long c);
void cipherLinearLayer(Ciphertext& cipher_res, Ciphertext& cipher_msg, double** weights, double* bias, Scheme_ &scheme, long w, long c_in, long c_out);

};

} // namespace heaan

#endif // SCHEME_H
