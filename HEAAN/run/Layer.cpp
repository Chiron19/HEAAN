#include "../src/HEAAN.h"

#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <set>
#include <string>
#include <iomanip>

using namespace std;
using namespace NTL;

namespace heaan {

static long CONV2D = 0;
static long CONV2DFAST = 1;
static long CONV2DFASTDOWNSAMPLE = 2;
static long BN = 3;
static long RELU = 4;
static long ADD = 5;
static long DOWNSAMPLE = 6;
static long AVGPOOL = 7;
static long LINEAR = 8;

double*** readKernals(string path, long c_out, long c_in) {
    ifstream file(path);
    string line;
    double*** kernals = new double**[c_out];
    for (int i = 0; i < c_out; i++) {
        kernals[i] = new double*[c_in];
        for (int j = 0; j < c_in; j++) {
            kernals[i][j] = new double[9];
            // each line is a 3x3 kernal, 9 double values, space separated
            getline(file, line);
            stringstream ss(line);
            for (int k = 0; k < 9; k++) {
                ss >> kernals[i][j][k];
            }
        }
    }
    cout << "kernals read done" << endl;
    return kernals;
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

void freeKernals(double*** kernals, long c_out, long c_in) {
    for (int i = 0; i < c_out; i++) {
        for (int j = 0; j < c_in; j++) {
            delete[] kernals[i][j];
        }
        delete[] kernals[i];
    }
    delete[] kernals;
}

void freeGammaBeta(double** gamma_beta, long c) {
    delete[] gamma_beta[0];
    delete[] gamma_beta[1];
    delete[] gamma_beta;
}

/**
 * @brief Determine the requirement of left and right rotation keys for each layer
 * 
 * @param leftRotKeys 
 * @param rightRotKeys 
 * @param params each param = (long[4]){type, w, c_in, c_out}
 */
void rotKeysRequirement(std::set<long>& leftRotKeys, std::set<long>& rightRotKeys, std::vector<std::array<long, 4>>& params) {
    for (auto& param : params) {
        long type = param[0];
        long w = param[1];
        long c_in = param[2];
        long c_out = param[3];
        if (type == CONV2D) {
            leftRotKeys.insert({1, w, w * w});
            rightRotKeys.insert({1, w, w * w});
        }
        else if (type == CONV2DFAST) {
            leftRotKeys.insert({1, w});
            for (long i = 1; i <= c_in / 2; i <<= 1) leftRotKeys.insert(i * w * w);
            rightRotKeys.insert(1);
            for (long i = 1; i < c_out; i++) rightRotKeys.insert(i * w * w);
        }
        else if (type == CONV2DFASTDOWNSAMPLE) {
            leftRotKeys.insert({1, w, 3 * w / 2, 3 * w * w / 4});
            for (long i = 1; i <= c_in / 2; i <<= 1) leftRotKeys.insert(i * w * w / 4);
            rightRotKeys.insert({1, w, w / 2});
            for (long i = 1; i < c_out; i++) rightRotKeys.insert(i * w * w / 4);
        }
        else if (type == DOWNSAMPLE) {
            leftRotKeys.insert({1, 3 * w / 2, 3 * w * w / 4});
            rightRotKeys.insert({1, w / 2, w * w / 4});
        }
        else if (type == AVGPOOL) {
            for (long i = w * w / 2; i > 0; i >>= 1) leftRotKeys.insert(i);
        }
        else if (type == LINEAR) {
            for (long i = 1; i <= c_in / 2; i <<= 1) leftRotKeys.insert(i * w * w);
            rightRotKeys.insert(w * w);
        }
    }
}

// Requirement: LeftRotKey: 1, w, w * w, RightRotKey: 1, w, w * w
void cipherConv2dLayer_wrapper(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c_in, long c_out, string path) {
    double*** kernals = readKernals(path, c_out, c_in);
    scheme.cipherConv2dLayer(cipher_res, cipher_msg, kernals, scheme, w, c_in, c_out);
    freeKernals(kernals, c_out, c_in);
}

// Requirement: LeftRotKey: 1, w, {1, 2, 4, ..., c_in / 2} * w * w, RightRotKey: 1, w, {1, ..., c_out - 1} * w * w
void cipherConv2dLayerFast_wrapper(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c_in, long c_out, string path) {
    double*** kernals = readKernals(path, c_out, c_in);
    scheme.cipherConv2dLayerFast(cipher_res, cipher_msg, kernals, scheme, w, c_in, c_out);
    freeKernals(kernals, c_out, c_in);
}

void cipherBatchNormLayer_wrapper(Ciphertext& cipher_res, Scheme_& scheme, long w, long c, string path) {
    double** gamma_beta = readGammaBeta(path, c);
    scheme.cipherBatchNormLayerAndEqual(cipher_res, scheme, w, c, gamma_beta[0], gamma_beta[1]);
    freeGammaBeta(gamma_beta, c);
}

// note: w and c are the width and channel of the image after downsampling
void cipherConv2dLayerFastDownsampling_wrapper(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c, string path) {
    double*** kernals = readKernals(path, c, c / 2);
    scheme.cipherConv2dLayerFastDownsampling(cipher_res, cipher_msg, kernals, scheme, w * 2, c / 2);
    cipher_res.n >>= 1;
    freeKernals(kernals, c, c / 2);
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
    bool skip = false;
    cout << "basicBlock start" << endl;
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    
    if (!SerializationUtils_::checkFile("./cipher/block.conv1.bin")) {
        cipherConv2dLayerFast_wrapper(cipher_res, cipher_msg, scheme, w, c, c, paths[0]);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/block.conv1.bin");
        cout << "conv1 done" << endl;
        cout << "cipher_res: " << cipher_res.n << ", " << cipher_res.logp << ", " << cipher_res.logq << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/block.bn1.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/block.conv1.bin");
            skip = false;
        }
        cipherBatchNormLayer_wrapper(cipher_res, scheme, w, c, paths[1]);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/block.bn1.bin");
        cout << "bn1 done" << endl;
        cout << "cipher_res: " << cipher_res.n << ", " << cipher_res.logp << ", " << cipher_res.logq << endl;
    } else {
        skip = true;
    }
    

    if (!SerializationUtils_::checkFile("./cipher/block.relu1.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/block.bn1.bin");
            skip = false;
        }
        scheme.cipherReLUAndEqual(cipher_res, scheme);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/block.relu1.bin");
        cout << "relu1 done" << endl;
        cout << "cipher_res: " << cipher_res.n << ", " << cipher_res.logp << ", " << cipher_res.logq << endl;
    } else {
        skip = true;
    }

    if (!SerializationUtils_::checkFile("./cipher/block.conv2.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/block.relu1.bin");
            skip = false;
        }
        Ciphertext cipher_temp(cipher_res);
        cipherConv2dLayerFast_wrapper(cipher_res, cipher_temp, scheme, w, c, c, paths[2]);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/block.conv2.bin");
        cipher_temp.free();
        cout << "conv2 done" << endl;
        cout << "cipher_res: " << cipher_res.n << ", " << cipher_res.logp << ", " << cipher_res.logq << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/block.bn2.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/block.conv2.bin");
            skip = false;
        }
        cipherBatchNormLayer_wrapper(cipher_res, scheme, w, c, paths[3]);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/block.bn2.bin");
        cout << "bn2 done" << endl;
        cout << "cipher_res: " << cipher_res.n << ", " << cipher_res.logp << ", " << cipher_res.logq << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/block.relu2.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/block.bn2.bin");
            skip = false;
        }
        scheme.addAndEqual(cipher_res, cipher_msg);
        scheme.cipherReLUAndEqual(cipher_res, scheme);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/block.relu2.bin");
        cout << "relu2 done" << endl;
        cout << "cipher_res: " << cipher_res.n << ", " << cipher_res.logp << ", " << cipher_res.logq << endl;
    }
    
    SerializationUtils_::deleteFile("./cipher/block.conv1.bin");
    SerializationUtils_::deleteFile("./cipher/block.bn1.bin");
    SerializationUtils_::deleteFile("./cipher/block.relu1.bin");
    SerializationUtils_::deleteFile("./cipher/block.conv2.bin");
    SerializationUtils_::deleteFile("./cipher/block.bn2.bin");
    SerializationUtils_::deleteFile("./cipher/block.relu2.bin");
    cout << "basicBlock end" << endl;
}

void downsamplingBlock(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, long w, long c, std::vector<string> paths) {
    bool skip = false;
    Ciphertext cipher_temp;
    cout << "downsamplingBlock start" << endl;
    cout << "cipher_msg: " << cipher_msg.n << ", " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.conv1.bin")) {
        cipherConv2dLayerFastDownsampling_wrapper(cipher_res, cipher_msg, scheme, w, c, paths[0]);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/downsample.conv1.bin");
        cout << "downsample.conv1 done" << endl;
        cout << "cipher_res: " << cipher_res.n << ", " << cipher_res.logp << ", " << cipher_res.logq << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.bn1.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/downsample.conv1.bin");
            skip = false;
        }   
        cipherBatchNormLayer_wrapper(cipher_res, scheme, w, c, paths[1]);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/downsample.bn1.bin");
        cout << "downsample.bn1 done" << endl;
        cout << "cipher_res: " << cipher_res.n << ", " << cipher_res.logp << ", " << cipher_res.logq << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.relu1.bin")) {
        if (skip) {     
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/downsample.bn1.bin");
            skip = false;
        }
        scheme.cipherReLUAndEqual(cipher_res, scheme);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/downsample.relu1.bin");
        cout << "downsample.relu1 done" << endl;
        cout << "cipher_res: " << cipher_res.n << ", " << cipher_res.logp << ", " << cipher_res.logq << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.conv2.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/downsample.relu1.bin");
            skip = false;
        }
        cipher_temp.copy(cipher_res);
        cipherConv2dLayerFast_wrapper(cipher_res, cipher_temp, scheme, w, c, c, paths[2]);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/downsample.conv2.bin");
        cout << "downsample.conv2 done" << endl;
        cout << "cipher_res: " << cipher_res.n << ", " << cipher_res.logp << ", " << cipher_res.logq << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.bn2.bin")) {   
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/downsample.conv2.bin");
            skip = false;
        }
        cipherBatchNormLayer_wrapper(cipher_res, scheme, w, c, paths[3]); 
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/downsample.bn2.bin");
        cout << "downsample.bn2 done" << endl;
        cout << "cipher_res: " << cipher_res.n << ", " << cipher_res.logp << ", " << cipher_res.logq << endl;
    } else {
        skip = true;    
    }
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.conv3.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/downsample.bn2.bin");
            skip = false;
        }
        cipherConv2dLayerFastDownsampling_wrapper(cipher_temp, cipher_msg, scheme, w, c, paths[4]);
        SerializationUtils_::writeCiphertext(cipher_temp, "./cipher/downsample.conv3.bin");
        cout << "downsample.conv3 done" << endl;
        cout << "cipher_temp: " << cipher_temp.n << ", " << cipher_temp.logp << ", " << cipher_temp.logq << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.bn3.bin")) {
        if (skip) {
            cipher_temp = *SerializationUtils_::readCiphertext("./cipher/downsample.conv3.bin");
            skip = false;
        }
        cipherBatchNormLayer_wrapper(cipher_temp, scheme, w, c, paths[5]);
        SerializationUtils_::writeCiphertext(cipher_temp, "./cipher/downsample.bn3.bin");
        cout << "downsample.bn3 done" << endl;
        cout << "cipher_temp: " << cipher_temp.n << ", " << cipher_temp.logp << ", " << cipher_temp.logq << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/downsample.relu2.bin")) {
        if (skip) {
            cipher_res = *SerializationUtils_::readCiphertext("./cipher/downsample.bn2.bin");
            cipher_temp = *SerializationUtils_::readCiphertext("./cipher/downsample.bn3.bin");
            skip = false;
        }
        scheme.addAndEqual(cipher_res, cipher_temp);
        scheme.cipherReLUAndEqual(cipher_res, scheme);
        SerializationUtils_::writeCiphertext(cipher_res, "./cipher/downsample.relu2.bin");
        cout << "downsample.relu2 done" << endl;
        cout << "cipher_res: " << cipher_res.n << ", " << cipher_res.logp << ", " << cipher_res.logq << endl;
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
    cout << "downsamplingBlock end" << endl;
}

void layerInit(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths) {
    cout << "layerInit start" << endl;
    cipherConv2dLayer_wrapper(cipher_res, cipher_msg, scheme, 32, 3, 16, paths[0]);
    SerializationUtils::writeCiphertext(cipher_res, "./cipher/layerInit.conv1.bin");
    cout << "layerInit.conv1 done" << endl;
    
    cipherBatchNormLayer_wrapper(cipher_res, scheme, 32, 16, paths[1]);
    SerializationUtils::writeCiphertext(cipher_res, "./cipher/layerInit.bn1.bin");
    cout << "layerInit.bn1 done" << endl;
    
    scheme.cipherReLUAndEqual(cipher_res, scheme);
    SerializationUtils::writeCiphertext(cipher_res, "./cipher/layerInit.relu1.bin");
    cout << "layerInit.relu1 done" << endl;
    
    cout << "layerInit end" << endl;
}

void layer1(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths) {
    bool skip = false;
    cout << "layer1 start" << endl;
    Ciphertext* cipher_temp = new Ciphertext[2];
    if (!SerializationUtils_::checkFile("./cipher/layer1.1.bin")) {
        basicBlock(cipher_temp[0], cipher_msg, scheme, 32, 16, {paths[2], paths[3], paths[4], paths[5]});
        SerializationUtils::writeCiphertext(cipher_temp[0], "./cipher/layer1.1.bin");
        cout << "layer1.1 done" << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/layer1.2.bin")) {
        if (skip) {
            cipher_temp[0] = *SerializationUtils_::readCiphertext("./cipher/layer1.1.bin");
            skip = false;
        }
        basicBlock(cipher_temp[1], cipher_temp[0], scheme, 32, 16, {paths[6], paths[7], paths[8], paths[9]});
        SerializationUtils::writeCiphertext(cipher_temp[1], "./cipher/layer1.2.bin");
        cout << "layer1.2 done" << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/layer1.3.bin")) {
        if (skip) {
            cipher_temp[1] = *SerializationUtils_::readCiphertext("./cipher/layer1.2.bin");
            skip = false;
        }
        basicBlock(cipher_res, cipher_temp[1], scheme, 32, 16, {paths[10], paths[11], paths[12], paths[13]});
        SerializationUtils::writeCiphertext(cipher_res, "./cipher/layer1.3.bin");
        cout << "layer1.3 done" << endl;
    } else {
        skip = true;
    }

    cipher_temp[0].free();
    cipher_temp[1].free();
    delete[] cipher_temp;
    
    cout << "layer1 end" << endl;
}

void layer2(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths) {
    bool skip = false;
    cout << "layer2 start" << endl;
    Ciphertext* cipher_temp = new Ciphertext[2];
    if (!SerializationUtils_::checkFile("./cipher/layer2.1.bin")) {
        downsamplingBlock(cipher_temp[0], cipher_msg, scheme, 16, 32, {paths[14], paths[15], paths[16], paths[17], paths[18], paths[19]});
        SerializationUtils::writeCiphertext(cipher_temp[0], "./cipher/layer2.1.bin");
        cout << "layer2.1 done" << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/layer2.2.bin")) {
        if (skip) {
            cipher_temp[0] = *SerializationUtils_::readCiphertext("./cipher/layer2.1.bin");
            skip = false;
        }
        basicBlock(cipher_temp[1], cipher_temp[0], scheme, 16, 32, {paths[20], paths[21], paths[22], paths[23]});
        SerializationUtils::writeCiphertext(cipher_temp[1], "./cipher/layer2.2.bin");
        cout << "layer2.2 done" << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/layer2.3.bin")) { 
        if (skip) {
            cipher_temp[1] = *SerializationUtils_::readCiphertext("./cipher/layer2.2.bin");
            skip = false;
        }
        basicBlock(cipher_res, cipher_temp[1], scheme, 16, 32, {paths[24], paths[25], paths[26], paths[27]});
        SerializationUtils::writeCiphertext(cipher_res, "./cipher/layer2.3.bin");
        cout << "layer2.3 done" << endl;
    } else {
        skip = true;
    }

    cipher_temp[0].free();
    cipher_temp[1].free();
    delete[] cipher_temp;
    
    cout << "layer2 end" << endl;
}

void layer3(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths) {
    bool skip = false;
    cout << "layer3 start" << endl;
    Ciphertext* cipher_temp = new Ciphertext[2];
    if (!SerializationUtils_::checkFile("./cipher/layer3.1.bin")) {
        downsamplingBlock(cipher_temp[0], cipher_msg, scheme, 8, 64, {paths[28], paths[29], paths[30], paths[31], paths[32], paths[33]});
        SerializationUtils::writeCiphertext(cipher_temp[0], "./cipher/layer3.1.bin");
        cout << "layer3.1 done" << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/layer3.2.bin")) {
        if (skip) {
            cipher_temp[0] = *SerializationUtils_::readCiphertext("./cipher/layer3.1.bin");
            skip = false;
        }
        basicBlock(cipher_temp[1], cipher_temp[0], scheme, 8, 64, {paths[34], paths[35], paths[36], paths[37]});
        SerializationUtils::writeCiphertext(cipher_temp[1], "./cipher/layer3.2.bin");
        cout << "layer3.2 done" << endl;
    } else {
        skip = true;
    }
    
    if (!SerializationUtils_::checkFile("./cipher/layer3.3.bin")) {
        if (skip) {
            cipher_temp[1] = *SerializationUtils_::readCiphertext("./cipher/layer3.2.bin");
            skip = false;
        }
        basicBlock(cipher_res, cipher_temp[1], scheme, 8, 64, {paths[38], paths[39], paths[40], paths[41]});
        SerializationUtils::writeCiphertext(cipher_res, "./cipher/layer3.3.bin");
        cout << "layer3.3 done" << endl;
    } else {
        skip = true;
    }

    cipher_temp[0].free();
    cipher_temp[1].free();
    delete[] cipher_temp;
    
    cout << "layer3 end" << endl;
}

void layerEnd(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme_& scheme, std::vector<string>& paths) {
    cout << "layerEnd start" << endl;
    scheme.cipherAvgPoolingAndEqual(cipher_msg, scheme, 8, 64);
    SerializationUtils::writeCiphertext(cipher_msg, "./cipher/layerEnd.avgpool.bin");
    cout << "layerEnd.avgpool done" << endl;
    
    cipherLinearLayer_wrapper(cipher_res, cipher_msg, scheme, 8, 64, 10, paths[42], paths[43]);
    SerializationUtils::writeCiphertext(cipher_res, "./cipher/layerEnd.linear.bin");
    cout << "layerEnd.linear done" << endl;
    
    cout << "layerEnd end" << endl;
}

} // namespace heaan