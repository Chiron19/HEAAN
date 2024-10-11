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

Ciphertext Cipher_SiLU(Ciphertext &cipher, Scheme &scheme, long logp, long logq, long n) {
    // f(x) = x * sigmoid(x)
    Ciphertext ctemp, cexp, cinv, csigmoid;

    SchemeAlgo schemeAlgo(scheme);
    schemeAlgo.function(csigmoid, cipher, SIGMOID, logq, 2);
    cout << "csigmoid " << endl;

    scheme.multAndEqual(csigmoid, cipher);
    cout << "csigmoid * cipher " << endl;
    // scheme.reScaleByAndEqual(cipher, logp);
    return csigmoid;
}

int main(int argc, char **argv) {

	srand(time(NULL));
	SetNumThreads(1);
	TimeUtils timeutils;
	Ring ring;
    SecretKey secretKey(ring);
    Scheme scheme(secretKey, ring);

    // Parameters //
    long logq = 300; ///< Ciphertext modulus (this value should be <= logQ in "scr/Params.h")
    long logp = 30; ///< Scaling Factor (larger logp will give you more accurate value)
    long logn = 15; ///< number of slot is 2^logn (this value should be < logN in "src/Params.h")
    long n = 1 << logn;
    long slots = n;
    long numThread = 8;

    double* vec1 = new double[100]; // Dynamically allocate memory for the array
    for (int i = 0; i < 100; i++) {
        vec1[i] = (i-50) / 10.0;
    }
    StringUtils::showVec(vec1, 100);

    complex<double>* mvec1 = new complex<double>[n]; // Dynamically allocate memory for the array
    for (int i = 0; i < 100; i++) {
        mvec1[i] = complex<double>(vec1[i], 0);
    }
    for (int i = 100; i < n; i++) {
        mvec1[i] = complex<double>(0, 0);
    }

    Ciphertext cipher;
    scheme.encrypt(cipher, mvec1, n, logp, logq);
    Ciphertext cipher_silu = Cipher_SiLU(cipher, scheme, logp, logq, n);

    complex<double>* dec0 = scheme.decrypt(secretKey, cipher_silu);
    for (int i = 0; i < 100; i++) {
        cout << dec0[i].real() << " ";
    }
    cout << endl;
    // StringUtils::showVec(dec0, 100);
    
	return 0;
}
