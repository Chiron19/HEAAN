#ifndef LAYER_CPP
#define LAYER_CPP

#include "../src/HEAAN.h"

#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <set>
#include <string>
#include <iomanip>

#include "Scheme.cpp"
#include "SerializationUtils.cpp"

using namespace std;
using namespace NTL;

namespace heaan {

double*** readkernels(string path, long c_out, long c_in) {
    ifstream file(path);
    string line;
    double*** kernels = new double**[c_out];
    for (int i = 0; i < c_out; i++) {
        kernels[i] = new double*[c_in];
        for (int j = 0; j < c_in; j++) {
            kernels[i][j] = new double[9];
            // each line is a 3x3 kernel, 9 double values, space separated
            getline(file, line);
            stringstream ss(line);
            for (int k = 0; k < 9; k++) {
                ss >> kernels[i][j][k];
            }
        }
    }
    return kernels;
}

double** readkernels1x1(string path, long c_out, long c_in) {
    ifstream file(path);
    string line;
    double** kernels = new double*[c_out];
    for (int i = 0; i < c_out; i++) {
        kernels[i] = new double[c_in];
        getline(file, line);
        stringstream ss(line);
        for (int j = 0; j < c_in; j++) {
            ss >> kernels[i][j];
        }
    }
    return kernels;
}

double** readGammaBeta(string path, long c) {
    ifstream file(path);
    string line;
    double** params = new double*[2];
    for (int i = 0; i < 2; i++) {
        params[i] = new double[c];
    }
    
    getline(file, line);
    stringstream ss(line);
    for (int k = 0; k < c; k++) {
        ss >> params[0][k];
    }
    getline(file, line);
    ss.clear(); // Clear the state of the stringstream
    ss.str(line); // Set the new line to the stringstream
    for (int k = 0; k < c; k++) {
        ss >> params[1][k];
    }
    return params;
}

double** readWeights(string path, long c_in, long c_out) {
    ifstream file(path);
    string line;
    double** weights = new double*[c_out];
    for (int i = 0; i < c_out; i++) {
        weights[i] = new double[c_in];
        getline(file, line);
        stringstream ss(line);
        for (int j = 0; j < c_in; j++) {
            ss >> weights[i][j];
        }
    }
    return weights;
}

double* readBias(string path, long c) {
    ifstream file(path);
    string line;
    double* bias = new double[c];
    getline(file, line);
    stringstream ss(line);
    for (int k = 0; k < c; k++) {
        ss >> bias[k];
    }
    return bias;
}

void freeWeights(double** weights, long c_in, long c_out) {
    for (int i = 0; i < c_out; i++) {
        delete[] weights[i];
    }
    delete[] weights;
}

void freeBias(double* bias, long c) {
    delete[] bias;
}

void freekernels(double*** kernels, long c_out, long c_in) {
    for (int i = 0; i < c_out; i++) {
        for (int j = 0; j < c_in; j++) {
            delete[] kernels[i][j];
        }
        delete[] kernels[i];
    }
    delete[] kernels;
}

void freekernels1x1(double** kernels, long c_out, long c_in) {
    delete[] kernels;
}

void freeGammaBeta(double** gamma_beta, long c) {
    delete[] gamma_beta[0];
    delete[] gamma_beta[1];
    delete[] gamma_beta;
}

// Requirement: LeftRotKey: 1, w, w * w, RightRotKey: 1, w, w * w
void cipherConv2dLayer_wrapper(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c_in, long c_out, string path) {
    double*** kernels = readkernels(path, c_out, c_in);
    scheme.cipherConv2dLayer(cipher_res, cipher_msg, kernels, scheme, w, c_in, c_out);
    freekernels(kernels, c_out, c_in);
}

// Requirement: LeftRotKey: 1, w, {1, 2, 4, ..., c_in / 2} * w * w, RightRotKey: 1, w, {1, ..., c_out - 1} * w * w
void cipherConv2dLayerFast_wrapper(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c_in, long c_out, string path) {
    double*** kernels = readkernels(path, c_out, c_in);
    scheme.cipherConv2dLayerFast(cipher_res, cipher_msg, kernels, scheme, w, c_in, c_out);
    freekernels(kernels, c_out, c_in);
}

void cipherBatchNormLayer_wrapper(Ciphertext& cipher_res, Scheme_& scheme, long w, long c, string path) {
    double** gamma_beta = readGammaBeta(path, c);
    scheme.cipherBatchNormLayerAndEqual(cipher_res, scheme, w, c, gamma_beta[0], gamma_beta[1]);
    freeGammaBeta(gamma_beta, c);
}

// note: w and c are the width and channel of the image after downsampling
void cipherConv2dLayerFastDownsampling_wrapper(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c, string path) {
    double*** kernels = readkernels(path, c, c / 2);
    scheme.cipherConv2dLayerFastDownsampling(cipher_res, cipher_msg, kernels, scheme, w * 2, c / 2);
    cipher_res.n >>= 1;
    freekernels(kernels, c, c / 2);
}

void cipherConv2d1x1LayerFastDownsampling_wrapper(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c, string path) {
    double** kernels = readkernels1x1(path, c, c / 2);
    scheme.cipherConv2d1x1LayerFastDownsampling(cipher_res, cipher_msg, kernels, scheme, w * 2, c / 2);
    cipher_res.n >>= 1;
    freekernels1x1(kernels, c, c / 2);
}

// Requirement: LeftRotKey: w * w, RightRotKey: w * w
void cipherLinearLayer_wrapper(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c_in, long c_out, string path_weight, string path_bias) {
    double** weights = readWeights(path_weight, c_in, c_out);
    double* bias = readBias(path_bias, c_out);
    scheme.cipherLinearLayer(cipher_res, cipher_msg, weights, bias, scheme, w, c_in, c_out);
    freeWeights(weights, c_in, c_out);
    freeBias(bias, c_out);
}


void basicBlock(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c, std::vector<string> paths) {
    TimeUtils timeutils, timeutils_;
    timeutils.start("basicBlock");
    bool skip = false;
    
    if (!SerializationUtils_::checkFile("./cipher/block.conv1.bin")) {
        timeutils_.start("block.conv1");
        cipherConv2dLayerFast_wrapper(cipher_res, cipher_msg, scheme, w, c, c, paths[0]);
        timeutils_.stop("block.conv1");
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/block.conv1.bin");
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/block.bn1.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/block.conv1.bin");
            skip = false;
        }
        timeutils_.start("block.bn1");
        cipherBatchNormLayer_wrapper(cipher_res, scheme, w, c, paths[1]);
        timeutils_.stop("block.bn1");
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/block.bn1.bin");
    } else {
        skip = true;
    }
    

    if (!SerializationUtils_::checkFile("./cipher/block.relu1.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/block.bn1.bin");
            skip = false;
        }
        timeutils_.start("block.relu1");
        scheme.cipherReLUAndEqual(cipher_res, scheme, 2);
        timeutils_.stop("block.relu1");
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/block.relu1.bin");
    } else {
        skip = true;
    }

    if (!SerializationUtils_::checkFile("./cipher/block.conv2.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/block.relu1.bin");
            skip = false;
        }
        timeutils_.start("block.conv2");
        Ciphertext cipher_temp(cipher_res);
        cipherConv2dLayerFast_wrapper(cipher_res, cipher_temp, scheme, w, c, c, paths[2]);
        timeutils_.stop("block.conv2");
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/block.conv2.bin");
        cipher_temp.free();
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/block.bn2.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/block.conv2.bin");
            skip = false;
        }
        timeutils_.start("block.bn2");
        cipherBatchNormLayer_wrapper(cipher_res, scheme, w, c, paths[3]);
        timeutils_.stop("block.bn2");
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/block.bn2.bin");
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/block.relu2.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/block.bn2.bin");
            skip = false;
        }
        timeutils_.start("block.relu2");
        scheme.reScaleToAndEqual(cipher_msg, cipher_res.logq);
        scheme.addAndEqual(cipher_res, cipher_msg);
        scheme.cipherReLUAndEqual(cipher_res, scheme, 2);
        timeutils_.stop("block.relu2");
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/block.relu2.bin");
    }
    
    SerializationUtils_::deleteFile("./cipher/block.conv1.bin");
    SerializationUtils_::deleteFile("./cipher/block.bn1.bin");
    SerializationUtils_::deleteFile("./cipher/block.relu1.bin");
    SerializationUtils_::deleteFile("./cipher/block.conv2.bin");
    SerializationUtils_::deleteFile("./cipher/block.bn2.bin");
    SerializationUtils_::deleteFile("./cipher/block.relu2.bin");
    timeutils.stop("basicBlock");
}

void downsamplingBlock(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c, std::vector<string> paths) {
    bool skip = false;
    Ciphertext cipher_temp;
    TimeUtils timeutils, timeutils_;
    timeutils.start("downsamplingBlock");
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.conv1.bin")) {
        timeutils_.start("downsample.conv1");
        cipherConv2dLayerFastDownsampling_wrapper(cipher_res, cipher_msg, scheme, w, c, paths[0]);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/downsample.conv1.bin");
        timeutils_.stop("downsample.conv1");
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.bn1.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/downsample.conv1.bin");
            skip = false;
        }
        timeutils_.start("downsample.bn1");
        cipherBatchNormLayer_wrapper(cipher_res, scheme, w, c, paths[1]);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/downsample.bn1.bin");
        timeutils_.stop("downsample.bn1");
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.relu1.bin")) {
        if (skip) {     
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/downsample.bn1.bin");
            skip = false;
        }
        timeutils_.start("downsample.relu1");
        scheme.cipherReLUAndEqual(cipher_res, scheme, 2);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/downsample.relu1.bin");
        timeutils_.stop("downsample.relu1");
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.conv2.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/downsample.relu1.bin");
            skip = false;
        }
        timeutils_.start("downsample.conv2");
        cipher_temp.copy(cipher_res);
        cipherConv2dLayerFast_wrapper(cipher_res, cipher_temp, scheme, w, c, c, paths[2]);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/downsample.conv2.bin");
        timeutils_.stop("downsample.conv2");
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.bn2.bin")) {   
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/downsample.conv2.bin");
            skip = false;
        }
        timeutils_.start("downsample.bn2");
        cipherBatchNormLayer_wrapper(cipher_res, scheme, w, c, paths[3]); 
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/downsample.bn2.bin");
        timeutils_.stop("downsample.bn2");
    } else {
        skip = true;    
    }
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.conv3.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/downsample.bn2.bin");
            skip = false;
        }
        timeutils_.start("downsample.conv3");
        cipherConv2d1x1LayerFastDownsampling_wrapper(cipher_temp, cipher_msg, scheme, w, c, paths[4]);
        SerializationUtils_::writeCiphertext(cipher_temp, "./cipher/downsample.conv3.bin");
        timeutils_.stop("downsample.conv3");
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.bn3.bin")) {
        if (skip) {
            cipher_temp = *SerializationUtils_::readCiphertext("./cipher/downsample.conv3.bin");
            skip = false;
        }
        timeutils_.start("downsample.bn3");
        cipherBatchNormLayer_wrapper(cipher_temp, scheme, w, c, paths[5]);
        SerializationUtils_::writeCiphertext(cipher_temp, "./cipher/downsample.bn3.bin");
        timeutils_.stop("downsample.bn3");
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.relu2.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/downsample.bn2.bin");
            cipher_temp = *SerializationUtils_::readCiphertext("./cipher/downsample.bn3.bin");
            skip = false;
        }
        timeutils_.start("downsample.relu2");
        scheme.reScaleToAndEqual(cipher_temp, cipher_res.logq);
        scheme.addAndEqual(cipher_res, cipher_temp);
        scheme.cipherReLUAndEqual(cipher_res, scheme, 2);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/downsample.relu2.bin");
        timeutils_.stop("downsample.relu2");
    }

    cipher_temp.free();
    SerializationUtils_::deleteFile("./cipher/downsample.conv1.bin");
    SerializationUtils_::deleteFile("./cipher/downsample.bn1.bin");
    SerializationUtils_::deleteFile("./cipher/downsample.relu1.bin");
    SerializationUtils_::deleteFile("./cipher/downsample.conv2.bin");
    SerializationUtils_::deleteFile("./cipher/downsample.bn2.bin");
    SerializationUtils_::deleteFile("./cipher/downsample.conv3.bin");
    SerializationUtils_::deleteFile("./cipher/downsample.bn3.bin");
    SerializationUtils_::deleteFile("./cipher/downsample.relu2.bin");
    timeutils.stop("downsamplingBlock");
}

void layerInit(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths) {
    bool skip = false;
    TimeUtils timeutils, timeutils_;
    timeutils.start("layerInit");
    
    if (!SerializationUtils_::checkFile("./cipher/layerInit.conv1.bin")) {
        timeutils_.start("layerInit.conv1");
        cipherConv2dLayer_wrapper(cipher_res, cipher_msg, scheme, 32, 3, 16, paths[0]);
        SerializationUtils::writeCiphertext(cipher_res, "./cipher/layerInit.conv1.bin");
        timeutils_.stop("layerInit.conv1");
    } else {
        skip = true;
    }

    if (!SerializationUtils_::checkFile("./cipher/layerInit.bn1.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/layerInit.conv1.bin");
        }
        timeutils_.start("layerInit.bn1");
        cipherBatchNormLayer_wrapper(cipher_res, scheme, 32, 16, paths[1]);
        SerializationUtils::writeCiphertext(cipher_res, "./cipher/layerInit.bn1.bin");
        timeutils_.stop("layerInit.bn1");

        timeutils_.start("layerInit.relu1");
        scheme.cipherReLUAndEqual(cipher_res, scheme, 2);
        SerializationUtils::writeCiphertext(cipher_res, "./cipher/layerInit.relu1.bin");
        timeutils_.stop("layerInit.relu1");
    }
    
    timeutils.stop("layerInit");
}

void layer1(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths) {
    bool skip = false;
    TimeUtils timeutils, timeutils_;
    timeutils.start("layer1");
    
    Ciphertext* cipher_temp = new Ciphertext[2];
    if (!SerializationUtils_::checkFile("./cipher/layer1.1.bin")) {
        timeutils_.start("layer1.1");
        basicBlock(cipher_temp[0], cipher_msg, scheme, 32, 16, {paths[2], paths[3], paths[4], paths[5]});
        SerializationUtils::writeCiphertext(cipher_temp[0], "./cipher/layer1.1.bin");
        timeutils_.stop("layer1.1");
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/layer1.2.bin")) {
        if (skip) {
            cipher_temp[0] = *SerializationUtils_::readCiphertext("./cipher/layer1.1.bin");
            skip = false;
        }
        timeutils_.start("layer1.2");
        basicBlock(cipher_temp[1], cipher_temp[0], scheme, 32, 16, {paths[6], paths[7], paths[8], paths[9]});
        SerializationUtils::writeCiphertext(cipher_temp[1], "./cipher/layer1.2.bin");
        timeutils_.stop("layer1.2");
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/layer1.3.bin")) {
        if (skip) {
            cipher_temp[1] = *SerializationUtils_::readCiphertext("./cipher/layer1.2.bin");
            skip = false;
        }
        timeutils_.start("layer1.3");
        basicBlock(cipher_res, cipher_temp[1], scheme, 32, 16, {paths[10], paths[11], paths[12], paths[13]});
        SerializationUtils::writeCiphertext(cipher_res, "./cipher/layer1.3.bin");
        timeutils_.stop("layer1.3");
    } else {
        skip = true;
    }

    cipher_temp[0].free();
    cipher_temp[1].free();
    delete[] cipher_temp;
    
    timeutils.stop("layer1");
}

void layer2(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths) {
    bool skip = false;
    TimeUtils timeutils, timeutils_;
    timeutils.start("layer2");

    Ciphertext* cipher_temp = new Ciphertext[2];
    if (!SerializationUtils_::checkFile("./cipher/layer2.1.bin")) {
        timeutils_.start("layer2.1");
        downsamplingBlock(cipher_temp[0], cipher_msg, scheme, 16, 32, {paths[14], paths[15], paths[16], paths[17], paths[18], paths[19]});
        SerializationUtils::writeCiphertext(cipher_temp[0], "./cipher/layer2.1.bin");
        timeutils_.stop("layer2.1");
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/layer2.2.bin")) {
        if (skip) {
            cipher_temp[0] = *SerializationUtils_::readCiphertext("./cipher/layer2.1.bin");
            skip = false;
        }
        timeutils_.start("layer2.2");
        basicBlock(cipher_temp[1], cipher_temp[0], scheme, 16, 32, {paths[20], paths[21], paths[22], paths[23]});
        SerializationUtils::writeCiphertext(cipher_temp[1], "./cipher/layer2.2.bin");
        timeutils_.stop("layer2.2");
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/layer2.3.bin")) { 
        if (skip) {
            cipher_temp[1] = *SerializationUtils_::readCiphertext("./cipher/layer2.2.bin");
            skip = false;
        }
        timeutils_.start("layer2.3");
        basicBlock(cipher_res, cipher_temp[1], scheme, 16, 32, {paths[24], paths[25], paths[26], paths[27]});
        SerializationUtils::writeCiphertext(cipher_res, "./cipher/layer2.3.bin");
        timeutils_.stop("layer2.3");
    } else {
        skip = true;
    }

    cipher_temp[0].free();
    cipher_temp[1].free();
    delete[] cipher_temp;

    timeutils.stop("layer2");
}

void layer3(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths) {
    bool skip = false;
    TimeUtils timeutils, timeutils_;
    timeutils.start("layer3");

    Ciphertext* cipher_temp = new Ciphertext[2];
    if (!SerializationUtils_::checkFile("./cipher/layer3.1.bin")) {
        timeutils_.start("layer3.1");
        downsamplingBlock(cipher_temp[0], cipher_msg, scheme, 8, 64, {paths[28], paths[29], paths[30], paths[31], paths[32], paths[33]});
        SerializationUtils::writeCiphertext(cipher_temp[0], "./cipher/layer3.1.bin");
        timeutils_.stop("layer3.1");
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/layer3.2.bin")) {
        if (skip) {
            cipher_temp[0] = *SerializationUtils_::readCiphertext("./cipher/layer3.1.bin");
            skip = false;
        }
        timeutils_.start("layer3.2");
        basicBlock(cipher_temp[1], cipher_temp[0], scheme, 8, 64, {paths[34], paths[35], paths[36], paths[37]});
        SerializationUtils::writeCiphertext(cipher_temp[1], "./cipher/layer3.2.bin");
        timeutils_.stop("layer3.2");
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/layer3.3.bin")) {
        if (skip) {
            cipher_temp[1] = *SerializationUtils_::readCiphertext("./cipher/layer3.2.bin");
            skip = false;
        }
        timeutils_.start("layer3.3");
        basicBlock(cipher_res, cipher_temp[1], scheme, 8, 64, {paths[38], paths[39], paths[40], paths[41]});
        SerializationUtils::writeCiphertext(cipher_res, "./cipher/layer3.3.bin");
        timeutils_.stop("layer3.3");
    } else {
        skip = true;
    }

    cipher_temp[0].free();
    cipher_temp[1].free();
    delete[] cipher_temp;

    timeutils.stop("layer3");
}

void layerEnd(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths) {
    TimeUtils timeutils, timeutils_;
    timeutils.start("layerEnd");
    
    timeutils_.start("layerEnd.avgpool");
    scheme.cipherAvgPoolingAndEqual(cipher_msg, scheme, 8, 64);
    SerializationUtils::writeCiphertext(cipher_msg, "./cipher/layerEnd.avgpool.bin");
    timeutils_.stop("layerEnd.avgpool");
    
    timeutils_.start("layerEnd.linear");
    cipherLinearLayer_wrapper(cipher_res, cipher_msg, scheme, 8, 64, 10, paths[42], paths[43]);
    SerializationUtils::writeCiphertext(cipher_res, "./cipher/layerEnd.linear.bin");
    timeutils_.stop("layerEnd.linear");

    timeutils.stop("layerEnd");
}

} // namespace heaan

#endif // !LAYER_CPP