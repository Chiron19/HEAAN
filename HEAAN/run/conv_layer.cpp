#include "../src/HEAAN.h"

#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <string>
#include <iomanip>

using namespace std;
using namespace NTL;
using namespace heaan;

/**
  * This file is for test HEAAN library
  * You can find more in src/TestScheme.h
  * "./TestHEAAN Encrypt" will run Encrypt Test
  * There are Encrypt, EncryptSingle, Add, Mult, iMult, RotateFast, Conjugate Tests
  */
void diagonal_packing(complex<double>* mvec, double* vec1, long row, long col, long n) {
    long ii = 0;
    for (long i = 0; i < row; i++) {
        for (long j = 0; j < col; j++) {
            mvec[ii++] = vec1[i * row + j];
        }
        for (long j = 0; j < col; j++) {
            mvec[ii++] = 0;
        }
    }
    for (long i = ii; i < n; i++) {
        mvec[i] = 0;
    }
}

void diagonal_packing_single_value(complex<double>* mvec, double v, long row, long col, long n) {
    long ii = 0;
    for (long i = 0; i < row; i++) {
        for (long j = 0; j < col; j++) {
            mvec[ii++] = v;
        }
        for (long j = 0; j < row; j++) {
            mvec[ii++] = 0;
        }
    }
    for (long i = ii; i < n; i++) {
        mvec[i] = 0;
    }
}

void print_diagonal_packing(complex<double>* mvec, long row, long col, long precision = 3L) {
    for (long i = 0; i < row * (row + col - 1); i++) {
        if (i && i % (row + col - 1) == 0) cout << endl;
        cout << setw(precision) << setprecision(precision) << round(mvec[i].real()) << " ";
    }
    cout << endl;
    cout << endl;
}

Ciphertext cipher_conv_3x3(Ciphertext& cipher_msg, double* kernal, Scheme scheme, long row, long col, long n, long logp, long logq) {
    Ciphertext cipher[9];
    Ciphertext cipher_rot[2];
    complex<double>** mvec = new complex<double>*[9]; // Dynamically allocate memory for the array
    for (long i = 0; i < 9; i++) {
        mvec[i] = new complex<double>[n];
        diagonal_packing_single_value(mvec[i], kernal[i], row, col, n);
        scheme.encrypt(cipher[i], mvec[i], n, logp, logq);
    }
    scheme.multAndEqual(cipher[4], cipher_msg);

    scheme.rightRotateFast(cipher_rot[0], cipher_msg, 1);
    scheme.multAndEqual(cipher[3], cipher_rot[0]);

    scheme.rightRotateFastAndEqual(cipher_rot[0], row + col);
    scheme.multAndEqual(cipher[0], cipher_rot[0]);

    scheme.leftRotateFastAndEqual(cipher_rot[0], 1);
    scheme.multAndEqual(cipher[1], cipher_rot[0]);

    scheme.leftRotateFastAndEqual(cipher_rot[0], 1);
    scheme.multAndEqual(cipher[2], cipher_rot[0]);

    scheme.leftRotateFast(cipher_rot[1], cipher_msg, 1);
    scheme.multAndEqual(cipher[5], cipher_rot[1]);

    scheme.leftRotateFastAndEqual(cipher_rot[1], row + col);
    scheme.multAndEqual(cipher[8], cipher_rot[1]);

    scheme.rightRotateFastAndEqual(cipher_rot[1], 1);
    scheme.multAndEqual(cipher[7], cipher_rot[1]);

    scheme.rightRotateFastAndEqual(cipher_rot[1], 1);
    scheme.multAndEqual(cipher[6], cipher_rot[1]);

    scheme.addAndEqual(cipher[0], cipher[1]);
    scheme.addAndEqual(cipher[2], cipher[3]);
    scheme.addAndEqual(cipher[5], cipher[6]);
    scheme.addAndEqual(cipher[7], cipher[8]);
    scheme.addAndEqual(cipher[0], cipher[2]);
    scheme.addAndEqual(cipher[5], cipher[7]);
    scheme.addAndEqual(cipher[4], cipher[5]);
    scheme.addAndEqual(cipher[0], cipher[4]);

    for (int i = 0; i < 9; i++) {
        delete[] mvec[i];
    }
    delete[] mvec;
    cipher_rot[0].free();
    cipher_rot[1].free();
    for (int i = 1; i < 9; i++) {
        cipher[i].free();
    }

    return cipher[0];
}

void cipher_resample(Ciphertext& cipher_msg, long channel_in, long channel_out, double* kernal, Scheme scheme, long row, long col, long n, long logp, long logq) {
    Ciphertext cipher_res;
    Ciphertext cipher_rot[2];
    complex<double>** mvec = new complex<double>*[9]; // Dynamically allocate memory for the array
    for (long i = 0; i < 9; i++) {
        mvec[i] = new complex<double>[n];
        diagonal_packing_single_value(mvec[i], kernal[i], row, col, n);
        scheme.encrypt(cipher[i], mvec[i], n, logp, logq);
    }
}

int main(int argc, char **argv) {

	srand(time(NULL));
	SetNumThreads(1);
	TimeUtils timeutils;
	Ring ring;
    SecretKey secretKey(ring);
    Scheme scheme(secretKey, ring);
    scheme.addLeftRotKey(secretKey, 1); ///< When you need left rotation for the vectorized message
    scheme.addLeftRotKey(secretKey, 10); ///< When you need left rotation for the vectorized message
    
    scheme.addRightRotKey(secretKey, 1); ///< When you need right rotation for the vectorized message
    scheme.addRightRotKey(secretKey, 10); ///< When you need left rotation for the vectorized message
    

    // Parameters //
    long logq = 300; ///< Ciphertext modulus (this value should be <= logQ in "scr/Params.h")
    long logp = 30; ///< Scaling Factor (larger logp will give you more accurate value)
    long logn = 15; ///< number of slot is 2^logn (this value should be < logN in "src/Params.h")
    long n = 1 << logn;
    long slots = n;
    long numThread = 8;

    double* vec1 = new double[25]; // Dynamically allocate memory for the array
    for (int i = 0; i < 25; i++) {
        vec1[i] = i + 1; // Assign a random value to each entry
    }

    complex<double>* mvec1 = new complex<double>[n]; // Dynamically allocate memory for the array
    diagonal_packing(mvec1, vec1, 5, 5, n);
    print_diagonal_packing(mvec1, 5, 5);
    cout << "done" << endl;

    double* vec2 = new double[9];
    for (int i = 0; i < 9; i++) {
        vec2[i] = i + 1;
    }

    Ciphertext cipher_msg;
    scheme.encrypt(cipher_msg, mvec1, n, logp, logq);
    Ciphertext cipher = cipher_conv_3x3(cipher_msg, vec2, scheme, 5, 5, n, logp, logq);

    complex<double>* dec0 = scheme.decrypt(secretKey, cipher);
    print_diagonal_packing(dec0, 5, 5);

    delete[] dec0;
    delete[] vec1;
    delete[] vec2;
    delete[] mvec1;

	return 0;
}
