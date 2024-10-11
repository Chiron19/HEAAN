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
        for (long j = 0; j < col; j++) {
            mvec[ii++] = 0;
        }
    }
    for (long i = ii; i < n; i++) {
        mvec[i] = 0;
    }
}

void print_diagonal_packing(complex<double>* mvec, long row, long col, long precision = 3) {
    for (long i = 0; i < row * (2 * col - 1); i++) {
        if (i && i % (2 * col - 1) == 0) cout << endl;
        cout << setw(precision) << setprecision(precision) << round(mvec[i].real()) << " ";
    }
    cout << endl;
    cout << endl;
}

int main(int argc, char **argv) {

	srand(time(NULL));
	SetNumThreads(1);
	TimeUtils timeutils;
	Ring ring;
    SecretKey secretKey(ring);
    Scheme scheme(secretKey, ring);
    scheme.addLeftRotKey(secretKey, 1); ///< When you need left rotation for the vectorized message
    scheme.addLeftRotKey(secretKey, 5); ///< When you need left rotation for the vectorized message
    scheme.addLeftRotKey(secretKey, 6); ///< When you need left rotation for the vectorized message
    scheme.addLeftRotKey(secretKey, 7); ///< When you need left rotation for the vectorized message
    
    scheme.addRightRotKey(secretKey, 1); ///< When you need right rotation for the vectorized message
    scheme.addRightRotKey(secretKey, 5); ///< When you need right rotation for the vectorized message
    scheme.addRightRotKey(secretKey, 6); ///< When you need left rotation for the vectorized message
    scheme.addRightRotKey(secretKey, 7); ///< When you need right rotation for the vectorized message


    // Parameters //
    long logq = 300; ///< Ciphertext modulus (this value should be <= logQ in "scr/Params.h")
    long logp = 30; ///< Scaling Factor (larger logp will give you more accurate value)
    long logn = 15; ///< number of slot is 2^logn (this value should be < logN in "src/Params.h")
    long n = 1 << logn;
    long slots = n;
    long numThread = 8;

    double* vec1 = new double[9]; // Dynamically allocate memory for the array
    for (int i = 0; i < 9; i++) {
        vec1[i] = i + 1; // Assign a random value to each entry
    }

    complex<double>* mvec1 = new complex<double>[n]; // Dynamically allocate memory for the array
    diagonal_packing(mvec1, vec1, 3, 3, n);
    print_diagonal_packing(mvec1, 3, 3);
    cout << "done" << endl;

    complex<double>** mvec = new complex<double>*[9]; // Dynamically allocate memory for the array
    for (long i = 0; i < 9; i++) {
        mvec[i] = new complex<double>[n];
        diagonal_packing_single_value(mvec[i], i+1, 3, 3, n);
    }
    cout << "done" << endl;

    Ciphertext cipher1;
    Ciphertext cipher2;
    Ciphertext cipher3;
    Ciphertext cipher4;
    Ciphertext cipher5;
    Ciphertext cipher6;
    Ciphertext cipher7;
    Ciphertext cipher8;
    Ciphertext cipher9;

    Ciphertext cipher[9];
    for (int i = 0; i < 9; i++) {
        scheme.encrypt(cipher[i], mvec[i], n, logp, logq);
    }
    cout << "encrypting" << endl;
	scheme.encrypt(cipher5, mvec1, n, logp, logq);
    
    scheme.rightRotateFast(cipher4, cipher5, 1);
    scheme.rightRotateFast(cipher3, cipher5, 5); // 2 * col - 1
    scheme.rightRotateFast(cipher2, cipher5, 6); // 2 * col
    scheme.rightRotateFast(cipher1, cipher5, 7); // 2 * col + 1

    scheme.leftRotateFast(cipher6, cipher5, 1);
    scheme.leftRotateFast(cipher7, cipher5, 5); // 2 * col - 1
    scheme.leftRotateFast(cipher8, cipher5, 6); // 2 * col
    scheme.leftRotateFast(cipher9, cipher5, 7); // 2 * col + 1


    scheme.multAndEqual(cipher[0], cipher1);
    scheme.multAndEqual(cipher[1], cipher2);
    scheme.multAndEqual(cipher[2], cipher3);
    scheme.multAndEqual(cipher[3], cipher4);
    scheme.multAndEqual(cipher[4], cipher5);
    scheme.multAndEqual(cipher[5], cipher6);
    scheme.multAndEqual(cipher[6], cipher7);
    scheme.multAndEqual(cipher[7], cipher8);
    scheme.multAndEqual(cipher[8], cipher9);
    
    scheme.addAndEqual(cipher[0], cipher[1]);
    scheme.addAndEqual(cipher[0], cipher[2]);
    scheme.addAndEqual(cipher[0], cipher[3]);
    scheme.addAndEqual(cipher[0], cipher[4]);
    scheme.addAndEqual(cipher[0], cipher[5]);
    scheme.addAndEqual(cipher[0], cipher[6]);
    scheme.addAndEqual(cipher[0], cipher[7]);
    scheme.addAndEqual(cipher[0], cipher[8]);

    complex<double>* dec0 = scheme.decrypt(secretKey, cipher[0]);
    print_diagonal_packing(dec0, 3, 3);
    
    // complex<double>* dec0 = scheme.decrypt(secretKey, cipher1);
    // print_diagonal_packing(dec0, 3, 3);

    // dec0 = scheme.decrypt(secretKey, cipher2);
    // print_diagonal_packing(dec0, 3, 3);

    // dec0 = scheme.decrypt(secretKey, cipher3);
    // print_diagonal_packing(dec0, 3, 3);

    // dec0 = scheme.decrypt(secretKey, cipher4);
    // print_diagonal_packing(dec0, 3, 3);

    // dec0 = scheme.decrypt(secretKey, cipher5);
    // print_diagonal_packing(dec0, 3, 3);

    // dec0 = scheme.decrypt(secretKey, cipher6);
    // print_diagonal_packing(dec0, 3, 3);

    // dec0 = scheme.decrypt(secretKey, cipher7);
    // print_diagonal_packing(dec0, 3, 3);

    // dec0 = scheme.decrypt(secretKey, cipher8);
    // print_diagonal_packing(dec0, 3, 3);

    // dec0 = scheme.decrypt(secretKey, cipher9);
    // print_diagonal_packing(dec0, 3, 3);

	return 0;
}
