#include "../src/HEAAN.h"

#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <string>
#include <iomanip>

using namespace std;
using namespace NTL;

namespace heaan {

class Scheme_: public Scheme {
public:

Scheme_(SecretKey& secretKey, Ring& ring, bool isSerialized = false) 
: Scheme(secretKey, ring, isSerialized) {}

void packKernal(complex<double>** mvec, double** kernal, long w, long c);
void packConst(complex<double>* mvec, long n, double* const_vec, long w, long c);
void packWeights(complex<double>* mvec, long n, double* weights, long w, long c);
void maskSlot(complex<double>* mvec, long n, long w, long c_start_id);
void cipherConv3x3(Ciphertext& cipher_res, Ciphertext& cipher_msg, double** kernal, Scheme& scheme, long w, long c);
void cipherChannelSumAndEqual(Ciphertext& cipher, Scheme& scheme, long w, long c, long c_target_id);
void cipherChannelFastSumAndEqual(Ciphertext& cipher, Scheme& scheme, long w, long c, long c_target_id);
void cipherConv2dLayer(Ciphertext &cipher_res, Ciphertext &cipher_msg, double*** kernals, Scheme &scheme, long w, long c_in, long c_out);
void cipherConv2dLayerFast(Ciphertext &cipher_res, Ciphertext &cipher_msg, double*** kernals, Scheme &scheme, long w, long c_in, long c_out);
void cipherConv2dLayerFastDownsampling(Ciphertext &cipher_res, Ciphertext &cipher_msg, double*** kernals, Scheme &scheme, long w, long c_in);
void cipherDownsamplingRow(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme& scheme, long w, long c);
void cipherDownsamplingColumn(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme& scheme, long w, long c);
void cipherDownsamplingChannel(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme& scheme, long w, long c);
void cipherDownsampling(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme& scheme, long w, long c);
void cipherBatchNormLayer(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme& scheme, long w, long c, double* gamma, double* beta);
void cipherBatchNormLayerAndEqual(Ciphertext& cipher_msg, Scheme& scheme, long w, long c, double* gamma, double* beta);
void cipherReLUAndEqual(Ciphertext& cipher_msg, Scheme& scheme);
void cipherAvgPoolingAndEqual(Ciphertext& cipher_msg, Scheme& scheme, long w, long c);
void cipherLinearLayer(Ciphertext& cipher_res, Ciphertext& cipher_msg, double** weights, double* bias, Scheme& scheme, long w, long c_in, long c_out);

};

/**
 * @brief Pack `kernel[c][9]`: 3x3 kernals of `c` channels to `mvec`
 * 
 * @param mvec 
 * @param kernal 
 * @param w 
 * @param c 
 */
void Scheme_::packKernal(complex<double>** mvec, double** kernal, long w, long c) {
    for (long i = 0; i < c; i++) {
        for (long j = 0; j < w; j++) {
            for (long k = 0; k < w; k++) {
                int ii = i * w * w + j * w + k;
                mvec[0][ii] = (j == 0 || k == 0) ? 0 : kernal[i][0];
                mvec[1][ii] = (j == 0) ? 0 : kernal[i][1];
                mvec[2][ii] = (j == 0 || k == w - 1) ? 0 : kernal[i][2];
                mvec[3][ii] = (k == 0) ? 0 : kernal[i][3];
                mvec[4][ii] = kernal[i][4];
                mvec[5][ii] = (k == w - 1) ? 0 : kernal[i][5];
                mvec[6][ii] = (j == w - 1 || k == 0) ? 0 : kernal[i][6];
                mvec[7][ii] = (j == w - 1) ? 0 : kernal[i][7];
                mvec[8][ii] = (j == w - 1 || k == w - 1) ? 0 : kernal[i][8];
            }
        }
    }
}

/**
 * @brief Pack `const_vec[c]`: constant vector of `c` channels to `mvec`. Each channel `i` is filled with `const_vec[i]`
 * 
 * @param mvec 
 * @param n 
 * @param const_vec 
 * @param w 
 * @param c 
 */
void Scheme_::packConst(complex<double>* mvec, long n, double* const_vec, long w, long c) {
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < w * w; j++) {
            mvec[i * w * w + j] = const_vec[i];
        }
    }
    for (int i = c * w * w; i < n; i++) {
        mvec[i] = 0;
    }
}

/**
 * @brief Pack `weights[c]`: weights of `c` channels to `mvec`, only first element of each channel `i` is `weight[i]`
 * 
 * @param mvec 
 * @param n 
 * @param weights 
 * @param w 
 * @param c 
 */
void Scheme_::packWeights(complex<double>* mvec, long n, double* weights, long w, long c) {
    for (int i = 0; i < n; i++) {
        mvec[i] = 0;
    }
    for (int i = 0; i < c; i++) {
        mvec[i * w * w] = weights[i];
    }
}

/**
 * @brief Mask slot of channel `c_start_id` of [0, `c`), equivalent to `packConst(const_vec={0, ..., 0, 1, 0, ..., 0})` 
 * 
 * @param mvec 
 * @param n 
 * @param w 
 * @param c_start_id 
 */
void Scheme_::maskSlot(complex<double>* mvec, long n, long w, long c_start_id) {
    for (long i = 0; i < c_start_id * w * w; i++) mvec[i] = 0;
    for (long i = c_start_id * w * w; i < (c_start_id+1) * w * w; i++) mvec[i] = 1;
    for (long i = (c_start_id+1) * w * w; i < n; i++) mvec[i] = 0;
}

/**
 * @brief Convolution operation with 3x3 kernal (padding (1, 1)) of c channels, image size w * w
 * 
 * @param cipher_res 
 * @param cipher_msg 
 * @param kernal 2d pointer to [c][9] 3x3 kernals of c channels
 * @param scheme 
 * @param w
 * @param c 
 * 
 * Requirement:
 * LeftRotKey: 1, w
 * RightRotKey: 1, w
 */
void Scheme_::cipherConv3x3(Ciphertext& cipher_res, Ciphertext& cipher_msg, double** kernal, Scheme& scheme, long w, long c) {
    long n = cipher_msg.n;
    long logp = cipher_msg.logp;
    long logq = cipher_msg.logq;
    Ciphertext* cipher = new Ciphertext[9];
    for (long i = 0; i < 9; i++) {
        cipher[i].copyParams(cipher_msg);
    }
    Ciphertext* cipher_rot = new Ciphertext[2];
    for (long i = 0; i < 2; i++) {
        cipher_rot[i].copy(cipher_msg);
    }
    complex<double>** mvec = new complex<double>*[9];
    for (long i = 0; i < 9; i++) {
        mvec[i] = new complex<double>[n];
    }
    packKernal(mvec, kernal, w, c);
   
    scheme.multByConstVec(cipher[4], cipher_rot[0], mvec[4], logp);

    scheme.rightRotateFastAndEqual(cipher_rot[0], 1);
    scheme.multByConstVec(cipher[3], cipher_rot[0], mvec[3], logp);

    scheme.rightRotateFastAndEqual(cipher_rot[0], w);
    scheme.multByConstVec(cipher[0], cipher_rot[0], mvec[0], logp);

    scheme.leftRotateFastAndEqual(cipher_rot[0], 1);
    scheme.multByConstVec(cipher[1], cipher_rot[0], mvec[1], logp);

    scheme.leftRotateFastAndEqual(cipher_rot[0], 1);
    scheme.multByConstVec(cipher[2], cipher_rot[0], mvec[2], logp);

    scheme.leftRotateFastAndEqual(cipher_rot[1], 1);
    scheme.multByConstVec(cipher[5], cipher_rot[1], mvec[5], logp);

    scheme.leftRotateFastAndEqual(cipher_rot[1], w);
    scheme.multByConstVec(cipher[8], cipher_rot[1], mvec[8], logp);

    scheme.rightRotateFastAndEqual(cipher_rot[1], 1);
    scheme.multByConstVec(cipher[7], cipher_rot[1], mvec[7], logp);

    scheme.rightRotateFastAndEqual(cipher_rot[1], 1);
    scheme.multByConstVec(cipher[6], cipher_rot[1], mvec[6], logp);

    scheme.addAndEqual(cipher[0], cipher[1]);
    scheme.addAndEqual(cipher[2], cipher[3]);
    scheme.addAndEqual(cipher[5], cipher[6]);
    scheme.addAndEqual(cipher[7], cipher[8]);
    scheme.addAndEqual(cipher[0], cipher[2]);
    scheme.addAndEqual(cipher[5], cipher[7]);
    scheme.addAndEqual(cipher[4], cipher[5]);
    scheme.addAndEqual(cipher[0], cipher[4]);

    cipher_res.copy(cipher[0]);
    scheme.reScaleByAndEqual(cipher_res, logp);
    cout << "cipherConv3x3 end: " << cipher_res.logp << ", " << cipher_res.logq << endl;

    for (int i = 0; i < 9; i++) {
        delete[] mvec[i];
    }
    delete[] mvec;
    for (int i = 0; i < 2; i++) {
        cipher_rot[i].free();
    }
    for (int i = 0; i < 9; i++) {
        cipher[i].free();
    }
    delete[] cipher_rot;
    delete[] cipher;
    cout << "convolution end" << endl;
}

/**
 * @brief Sum all `c` channels to channel `c_target_id` of [0, `c`) (in-place)
 * 
 * @param cipher 
 * @param scheme 
 * @param w 
 * @param c 
 * @param c_target_id 
 * 
 * Requirement:
 * LeftRotKey: w * w
 * RightRotKey: w * w
 * 
 */
void Scheme_::cipherChannelSumAndEqual(Ciphertext& cipher, Scheme& scheme, long w, long c, long c_target_id) {
    cout << "channel sum start" << endl;
    long n = cipher.n;
    long logp = cipher.logp;
    long logq = cipher.logq;
    Ciphertext cipher_rot(cipher);
    for (long i = 0; i < c_target_id; i++) {
        scheme.rightRotateFastAndEqual(cipher_rot, w * w);
    }
    for (long i = 1; i < c; i++) {
        scheme.leftRotateFastAndEqual(cipher_rot, w * w);
        cout << "channel sum rotate " << i << " done" << endl;
        scheme.addAndEqual(cipher, cipher_rot);
        cout << "channel sum add " << i << " done" << endl;
    }
    complex<double>* mvec = new complex<double>[n];
    maskSlot(mvec, n, w, c_target_id);
    scheme.multByConstVec(cipher, cipher_rot, mvec, logp);
    scheme.reScaleByAndEqual(cipher, logp);
    cipher_rot.free();
    delete[] mvec;
    cout << "channel sum end" << endl;
}

/**
 * @brief Sum all `c` channels to channel `c_target_id` of [0, `c`) (in-place)
 * 
 * @param cipher 
 * @param scheme 
 * @param w 
 * @param c 
 * @param c_target_id 
 * 
 * Requirement:
 * LeftRotKey: {1, 2, 4, ..., c / 2} * w * w
 * RightRotKey: c_target_id * w * w
 * 
 */
void Scheme_::cipherChannelFastSumAndEqual(Ciphertext& cipher, Scheme& scheme, long w, long c, long c_target_id) {
    cout << "channel sum start" << endl;
    long n = cipher.n;
    long logp = cipher.logp;
    long logq = cipher.logq;
    Ciphertext cipher_rot(cipher);
    Ciphertext cipher_temp;
    if (c_target_id) {
        scheme.rightRotateFastAndEqual(cipher_rot, c_target_id * w * w);
    }
    for (long i = c / 2; i > 0; i /= 2) {
        scheme.leftRotateFast(cipher_temp, cipher_rot, i * w * w);
        cout << "channel sum rotate " << i << " done" << endl;
        scheme.addAndEqual(cipher_rot, cipher_temp);
        cout << "channel sum add " << i << " done" << endl;
    }
    complex<double>* mvec = new complex<double>[n];
    maskSlot(mvec, n, w, c_target_id);
    scheme.multByConstVec(cipher, cipher_rot, mvec, logp);
    scheme.reScaleByAndEqual(cipher, logp);
    cipher_rot.free();
    cipher_temp.free();
    delete[] mvec;
    cout << "channel sum end" << endl;
}

/**
 * @brief Convolution Layer with 3x3 kernal (padding (1, 1)) of c channels, image size w * w
 * 
 * @param cipher_res 
 * @param cipher_msg 
 * @param kernals 3d pointer to [c_out][c_in][9] 3x3 kernals
 * @param scheme 
 * @param w 
 * @param c_in 
 * @param c_out 
 * 
 * Consumed Level: 2
 * 
 * Requirement:
 * LeftRotKey: 1, w, w * w
 * RightRotKey: 1, w, w * w
 * 
 */
void Scheme_::cipherConv2dLayer(Ciphertext &cipher_res, Ciphertext &cipher_msg, double*** kernals, Scheme &scheme, long w, long c_in, long c_out) {
    cout << "convolution start (LeftRotKey:" << 1 << ", " << w << ", " << w * w << ", RightRotKey:" << 1 << ", " << w << ", " << w * w << ")" << endl;
    long n = cipher_msg.n;
    long logp = cipher_msg.logp;
    long logq = cipher_msg.logq;
    cout << "cipherConv2dLayer start: " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    Ciphertext cipher_conv;
    for (long i = 0; i < c_out; i++)
    {
        cipherConv3x3(cipher_conv, cipher_msg, kernals[i], scheme, w, c_in);
        cout << "convolution channel_out " << i << " done" << endl;
        cipherChannelSumAndEqual(cipher_conv, scheme, w, c_in, i);
        cout << "channel sum channel_out " << i << " done" << endl;
        if (i == 0) {
            cipher_res.copy(cipher_conv);
        } else {
            scheme.addAndEqual(cipher_res, cipher_conv);
        }
        cout << "add channel_out " << i << " done" << endl;
    }
    
    cipher_conv.free();
    cout << "cipherConv2dLayer end: " << cipher_res.logp << ", " << cipher_res.logq << endl;
    cout << "convolution end" << endl;
}

/**
 * @brief Convolution Layer with 3x3 kernal (padding (1, 1)) of c channels, image size w * w
 * 
 * @param cipher_res 
 * @param cipher_msg 
 * @param kernals 3d pointer to [c_out][c_in][9] 3x3 kernals
 * @param scheme 
 * @param w 
 * @param c_in 
 * @param c_out 
 * 
 * Consumed Level: 2
 * 
 * Requirement:
 * LeftRotKey: 1, w, {1, 2, 4, ..., c_in / 2} * w * w 
 * RightRotKey: 1, w, {1, ..., c_out - 1} * w * w
 * 
 */
void Scheme_::cipherConv2dLayerFast(Ciphertext &cipher_res, Ciphertext &cipher_msg, double*** kernals, Scheme &scheme, long w, long c_in, long c_out) {
    cout << "convolution start (LeftRotKey:" <<  1 << ", " << w << ", ";
    for (long i = 1; i <= c_in / 2 ; i *= 2)
    {
        cout << i * w * w << ", ";
    }
    cout << " RightRotKey:" << 1 << ", " << w << ", ";
    for (long i = 1; i < c_out; i++)
    {
        cout << i * w * w << ", ";
    }
    cout << ")" << endl;
    long n = cipher_msg.n;
    long logp = cipher_msg.logp;
    long logq = cipher_msg.logq;
    cout << "cipherConv2dLayer start: " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    Ciphertext cipher_conv;
    for (long i = 0; i < c_out; i++)
    {
        cipherConv3x3(cipher_conv, cipher_msg, kernals[i], scheme, w, c_in);
        cout << "convolution channel_out " << i << " done" << endl;
        cipherChannelFastSumAndEqual(cipher_conv, scheme, w, c_in, i);
        cout << "channel sum channel_out " << i << " done" << endl;
        if (i == 0) {
            cipher_res.copy(cipher_conv);
        } else {
            scheme.addAndEqual(cipher_res, cipher_conv);
        }
        cout << "add channel_out " << i << " done" << endl;
    }
    
    cipher_conv.free();
    cout << "cipherConv2dLayer end: " << cipher_res.logp << ", " << cipher_res.logq << endl;
    cout << "convolution end" << endl;
}

/**
 * @brief Convolution Downsampling Layer with 3x3 kernal (padding (1, 1), stride (2, 2)) of c_in channels, image size w * w to (w / 2) * (w / 2)
 * 
 * @param cipher_res 
 * @param cipher_msg 
 * @param kernals 3d pointer to [c_out=c_in*2][c_in][9] 3x3 kernals
 * @param scheme 
 * @param w 
 * @param c_in 
 * 
 * Consumed Level: 2
 * 
 * Requirement:
 * LeftRotKey: 1, w, {1, 2, 4, ..., c_in / 2} * w * w / 4, 3 * w / 2, 3 * w * w / 4 
 * RightRotKey: 1, w, {1, ..., c_out - 1} * w * w / 4, w / 2
 * 
 */
void Scheme_::cipherConv2dLayerFastDownsampling(Ciphertext &cipher_res, Ciphertext &cipher_msg, double*** kernals, Scheme &scheme, long w, long c_in) {
    cout << "convolution downsampling start " << endl;
    long n = cipher_msg.n;
    long logp = cipher_msg.logp;
    long logq = cipher_msg.logq;
    cout << "cipherConv2dLayer start: " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    Ciphertext cipher_conv;
    Ciphertext* cipher_ds = new Ciphertext[2];
    for (long j = 0; j < 2; j++)
    {
        for (long i = 0; i < c_in; i++)
        {
            cipherConv3x3(cipher_conv, cipher_msg, kernals[j * c_in + i], scheme, w, c_in);
            cout << "convolution channel_out " << i << " done" << endl;
            cipherChannelFastSumAndEqual(cipher_conv, scheme, w, c_in, i);
            cout << "channel sum channel_out " << i << " done" << endl;
            if (i == 0) {
                cipher_res.copy(cipher_conv);
            } else {
                scheme.addAndEqual(cipher_res, cipher_conv);
            }
            cout << "add channel_out " << i << " done" << endl;
        }
        cipherDownsampling(cipher_ds[j], cipher_res, scheme, w, c_in);
    }
    
    cipher_res.copy(cipher_ds[0]);
    scheme.rightRotateFastAndEqual(cipher_ds[1], c_in * w * w / 4);
    scheme.addAndEqual(cipher_res, cipher_ds[1]);
    
    cipher_ds[0].free();
    cipher_ds[1].free();    
    cipher_conv.free();
    cout << "cipherConv2dLayer end: " << cipher_res.logp << ", " << cipher_res.logq << endl;
    cout << "convolution downsampling end" << endl;
}

/**
 * @brief Downsampling 1/2 by row, [c, w, w] -> [[c, w/2, w] [0 ... 0]]
 * 
 * @param cipher_res 
 * @param cipher_msg 
 * @param scheme 
 * @param w 
 * @param c 
 * 
 * Requirement:
 * LeftRotKey: 1
 * RightRotKey: 1
 */
void Scheme_::cipherDownsamplingRow(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme& scheme, long w, long c) {
    long n = cipher_msg.n;
    long logp = cipher_msg.logp;
    long logq = cipher_msg.logq;
    complex<double>* mask = new complex<double>[n];
    Ciphertext cipher_rot(cipher_msg);
    Ciphertext cipher_mask;
    for (int i = 0; i < n; i++) {
        mask[i] = 0;
    }
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < w; j+=2) {
            mask[i * w * w + j * w] = 1;
        }
    }
    scheme.encrypt(cipher_mask, mask, n, logp, logq);
    Ciphertext cipher_temp;
    scheme.mult(cipher_res, cipher_mask, cipher_rot);
    cout << cipher_rot.logp << ", " << cipher_rot.logq << endl;
    for (int i = 1; i < w / 2; i++) {
        scheme.leftRotateFastAndEqual(cipher_rot, 1);
        scheme.rightRotateFastAndEqual(cipher_mask, 1);
        scheme.mult(cipher_temp, cipher_mask, cipher_rot);
        scheme.addAndEqual(cipher_res, cipher_temp);
        cout << cipher_res.logp << ", " << cipher_res.logq << endl;
        cout << "downsampling row add " << i << " done" << endl;
    }

    scheme.reScaleByAndEqual(cipher_res, logp);

    delete[] mask;
    cipher_rot.free();
    cipher_mask.free();
    cipher_temp.free();
}

/**
 * @brief Downsampling 1/2 by column, [[c, w/2, w] [0 ... 0]] -> [[c, w/2, w/2] [0 ... 0]; [0 ... 0]]
 * 
 * @param cipher_res 
 * @param cipher_msg 
 * @param scheme 
 * @param w 
 * @param c 
 * 
 * Requirement:
 * LeftRotKey: 3 * w / 2
 * RightRotKey: w / 2
 * 
 */
void Scheme_::cipherDownsamplingColumn(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme& scheme, long w, long c) {
    long n = cipher_msg.n;
    long logp = cipher_msg.logp;
    long logq = cipher_msg.logq;
    complex<double>* mask = new complex<double>[n];
    Ciphertext cipher_rot(cipher_msg);
    Ciphertext cipher_mask;
    for (int i = 0; i < n; i++) {
        mask[i] = 0;
    }
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < w / 2; j++) {
            mask[i * w * w + j] = 1;
        }
    }
    scheme.encrypt(cipher_mask, mask, n, logp, logq);
    Ciphertext cipher_temp;
    scheme.mult(cipher_res, cipher_mask, cipher_rot);
    cout << cipher_rot.logp << ", " << cipher_rot.logq << endl;
    for (int i = 1; i < w / 2; i++) {
        scheme.leftRotateFastAndEqual(cipher_rot, 3 * w / 2);
        scheme.rightRotateFastAndEqual(cipher_mask, w / 2);
        scheme.mult(cipher_temp, cipher_mask, cipher_rot);
        scheme.addAndEqual(cipher_res, cipher_temp);
        cout << cipher_res.logp << ", " << cipher_res.logq << endl;
        cout << "downsampling column add " << i << " done" << endl;
    }
    scheme.reScaleByAndEqual(cipher_res, logp);
    delete[] mask;
    cipher_rot.free();
    cipher_mask.free();
    cipher_temp.free();
}

/**
 * @brief Downsampling 1/4 by channel, [c, w, w] -> [c, w/2, w/2] [0 ... 0]
 * 
 * @param cipher_res 
 * @param cipher_msg 
 * @param scheme 
 * @param w 
 * @param c 
 * 
 * Requirement:
 * LeftRotKey: 3 * w * w / 4
 * RightRotKey: w * w / 4
 * 
 */
void Scheme_::cipherDownsamplingChannel(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme& scheme, long w, long c) {
    long n = cipher_msg.n;
    long logp = cipher_msg.logp;
    long logq = cipher_msg.logq;
    complex<double>* mask = new complex<double>[n];
    Ciphertext cipher_rot(cipher_msg);
    Ciphertext cipher_mask;
    for (int i = 0; i < n; i++) {
        mask[i] = 0;
    }
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < w * w / 4; j++) {
            mask[i * w * w + j] = 1;
        }
    }
    scheme.encrypt(cipher_mask, mask, n, logp, logq);
    Ciphertext cipher_temp;
    scheme.mult(cipher_res, cipher_mask, cipher_rot);
    cout << cipher_rot.logp << ", " << cipher_rot.logq << endl;
    for (int i = 1; i < c; i++) {
        scheme.leftRotateFastAndEqual(cipher_rot, 3 * w * w / 4);
        scheme.rightRotateFastAndEqual(cipher_mask, w * w / 4);
        scheme.mult(cipher_temp, cipher_mask, cipher_rot);
        scheme.addAndEqual(cipher_res, cipher_temp);
        cout << cipher_res.logp << ", " << cipher_res.logq << endl;
        cout << "downsampling channel add " << i << " done" << endl;
    }
    scheme.reScaleByAndEqual(cipher_res, logp);
    delete[] mask;
    cipher_rot.free();
    cipher_mask.free();
    cipher_temp.free();
}

/**
 * @brief Downsampling in order by row, column, and channel, [c, w, w] -> [c, w/2, w/2]
 * 
 * @param cipher_res 
 * @param cipher_msg 
 * @param scheme 
 * @param w 
 * @param c 
 * 
 * Consumed Level: 3
 * 
 * Requirement:
 * LeftRotKey: 1, 3 * w / 2, 3 * w * w / 4
 * RightRotKey: 1, w / 2, w * w / 4
 */
void Scheme_::cipherDownsampling(Ciphertext& cipher_res, Ciphertext& cipher_msg, Scheme& scheme, long w, long c) {
    cout << "downsampling start (LeftRotKey:" << 1 << ", " << 3 * w / 2 << ", " << 3 * w * w / 4 << ", RightRotKey:" << 1 << ", " << w / 2 << ", " << w * w / 4 << ")" << endl;
    Ciphertext cipher_ds_1;
    Ciphertext cipher_ds_2;

    cipherDownsamplingRow(cipher_ds_1, cipher_msg, scheme, w, c);
    cout << "downsampling row done" << cipher_ds_1.logp << ", " << cipher_ds_1.logq << endl;
    cipherDownsamplingColumn(cipher_ds_2, cipher_ds_1, scheme, w, c);
    cout << "downsampling column done" << cipher_ds_2.logp << ", " << cipher_ds_2.logq << endl;
    cipherDownsamplingChannel(cipher_res, cipher_ds_2, scheme, w, c);
    cout << "downsampling channel done" << cipher_res.logp << ", " << cipher_res.logq << endl;
    cipher_ds_1.free();
    cipher_ds_2.free();
}

/**
 * @brief Batch Normalization Layer
 * 
 * @param cipher_msg 
 * @param scheme 
 * @param w 
 * @param c 
 * @param gamma 
 * @param beta 
 * 
 * Consumed Level: 1
 * 
 * For each channel i, Y = gamma[i] * X + beta[i]
 */
void Scheme_::cipherBatchNormLayerAndEqual(Ciphertext& cipher_msg, Scheme& scheme, long w, long c, double* gamma, double* beta) {
    long n = cipher_msg.n;
    long logp = cipher_msg.logp;
    long logq = cipher_msg.logq;
    cout << "cipherBatchNormLayerAndEqual start: " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    complex<double>* mvec_a = new complex<double>[n];
    complex<double>* mvec_b = new complex<double>[n];
    packConst(mvec_a, n, gamma, w, c);
    packConst(mvec_b, n, beta, w, c);
    
    scheme.multByConstVecAndEqual(cipher_msg, mvec_a, logp);
    scheme.reScaleByAndEqual(cipher_msg, logp);
    scheme.addConstAndEqual(cipher_msg, *mvec_b, logp);
    cout << "cipherBatchNormLayerAndEqual end: " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    delete[] mvec_a;
    delete[] mvec_b;
}

/**
 * @brief ReLU Activation Layer with polynomial approximation
 * 
 * @param cipher_msg 
 * @param scheme 
 * 
 * Consumed Level: 1
 * 
 * f(x) = x * sigmoid(x)
 */
void Scheme_::cipherReLUAndEqual(Ciphertext& cipher_msg, Scheme& scheme) {
    long logp = cipher_msg.logp;
    scheme.squareAndEqual(cipher_msg);
    scheme.reScaleByAndEqual(cipher_msg, logp);
    // Ciphertext cipher_relu;
    // cipher_relu.copyParams(cipher_msg);
    // SchemeAlgo schemeAlgo(scheme);
    // long logp = cipher_msg.logp;
    // schemeAlgo.function(cipher_relu, cipher_msg, SIGMOID, logp, 4);
    // scheme.multAndEqual(cipher_relu, cipher_msg);
    // cipher_msg.copy(cipher_relu);
    // scheme.reScaleByAndEqual(cipher_msg, logp);
    // cout << "cipherReLUAndEqual end: " << cipher_msg.logp << ", " << cipher_msg.logq << endl;
    // cipher_relu.free();
}

/**
 * @brief Average Pooling Layer, keep the average of channel at first element of each channel
 * 
 * @param cipher_msg 
 * @param scheme 
 * @param w 
 * @param c 
 * 
 * Requirement:
 * LeftRotKey: (w * w) >> 1 ... log2(w * w)
 * 
 */
void Scheme_::cipherAvgPoolingAndEqual(Ciphertext& cipher_msg, Scheme& scheme, long w, long c) {
    cout << "avg pooling start " << endl;
    long logp = cipher_msg.logp;
    Ciphertext cipher_temp;
    for (int i = w * w / 2; i > 0; i /= 2) {
        cipher_temp.copy(cipher_msg);
        scheme.leftRotateFastAndEqual(cipher_temp, i);
        scheme.addAndEqual(cipher_msg, cipher_temp);
        cout << "avg pooling add " << i << " done" << endl;
    }
}

/**
 * @brief Linear Layer, keep the linear combination of weights at first element of each channel of [0, `c_out`).
 * 
 * @param cipher_res 
 * @param cipher_msg 
 * @param weights 2d pointer to [c_out][c_in] weights
 * @param bias 1d pointer to [c_out] bias
 * @param scheme 
 * @param w 
 * @param c_in 
 * @param c_out 
 * 
 * For each channel i, Y[i] = sum(weights[i][j] * X[j]) + bias[i]
 * 
 * Consumed Level: 2
 * 
 * Requirement:
 * LeftRotKey: {1, 2, 4, ..., c_in / 2} * w * w
 * RightRotKey: w * w
 */
void Scheme_::cipherLinearLayer(Ciphertext& cipher_res, Ciphertext& cipher_msg, double** weights, double* bias, Scheme& scheme, long w, long c_in, long c_out) {
    cout << "linear layer start " << endl;
    long n = cipher_msg.n;
    long logp = cipher_msg.logp;
    long logq = cipher_msg.logq;
    complex<double>* mvec = new complex<double>[n];
    Ciphertext cipher_temp;
    Ciphertext cipher_sum;
    for (long i = 0; i < c_out; i++) {
        cipher_temp.copy(cipher_msg);
        packWeights(mvec, n, weights[i], w, c_in);
        scheme.multByConstVecAndEqual(cipher_temp, mvec, logp);
        scheme.reScaleByAndEqual(cipher_temp, logp);

        cout << "linear layer mult " << i << " done" << endl;

        for (int j = 1; j < i; j++) {
            scheme.rightRotateFastAndEqual(cipher_temp, w * w);
        }
        cipher_sum.copy(cipher_temp);
        for (int j = c_in / 2; j > 0; j >>= 1) {
            scheme.leftRotateFastAndEqual(cipher_temp, j * w * w);
            scheme.addAndEqual(cipher_sum, cipher_temp);
            cipher_temp.copy(cipher_sum);
        }
        maskSlot(mvec, n, w, i);
        scheme.multByConstVecAndEqual(cipher_sum, mvec, logp);
        scheme.reScaleByAndEqual(cipher_sum, logp);
        if (i == 0) {
            cipher_res.copy(cipher_sum);
        } else {
            scheme.addAndEqual(cipher_res, cipher_sum);
        }

        cout << "linear layer add " << i << " done" << endl;
    }
    packConst(mvec, n, bias, w, c_out);
    scheme.encrypt(cipher_temp, mvec, n, logp, logq);
    scheme.addAndEqual(cipher_res, cipher_temp);
    cipher_temp.free();
    cipher_sum.free();
    delete[] mvec;
}

} // namespace heaan