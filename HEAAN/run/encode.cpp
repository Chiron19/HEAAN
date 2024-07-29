/*
* Copyright (c) by CryptoLab inc.
* This program is licensed under a
* Creative Commons Attribution-NonCommercial 3.0 Unported License.
* You should have received a copy of the license along with this
* work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.
*/

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
int main(int argc, char **argv) {
	ofstream cout("output.txt");

	long logq; ///< Ciphertext Modulus q = 2^logq
	long logp; ///< Real message will be quantized by multiplying 2^logp
	long logn; ///< The number of slots n = 2^logn

	logq = 155, logp = 30, logn = 0;
	// cout << "(logq, logp, logn)=" << "(" << logq << ", " << logp << ", " << logn << ")" << endl;
	// TestScheme::testEncode(logq, logp, logn);
	// TestScheme::testEncrypt(logq, logp, logn);
	// TestScheme::testEncodeSingle(logq, logp);

	srand(time(NULL));
	SetNumThreads(1);
	TimeUtils timeutils;
	Ring ring;
	SecretKey secretKey(ring);
	Scheme scheme(secretKey, ring);

	long n = (1 << logn);
	complex<double>* mvec = EvaluatorUtils::randomComplexArray(n);
	// complex<double>* mvec = new complex<double>[n];
	Plaintext plain(logp, logq, n);
	Ciphertext cipher;

	long slots = plain.n;
	long gap = Nh / slots;

	// timeutils.start("Encode");
	// scheme.encode(plain, mvec, n, logp, logq);
	complex<double>* uvals = new complex<double>[slots];
	copy(mvec, mvec + slots, uvals);
	int idx, jdx;

	ring.EMBInv(uvals, slots);

	for (int i = 0, jdx = Nh, idx = 0; i < slots; ++i, jdx += gap, idx += gap) {
		// cout << "uval[" << i << "]: " << uvals[i] << endl;
		plain.mx[idx] = EvaluatorUtils::scaleUpToZZ(uvals[i].real(), plain.logp  + logQ);
		plain.mx[jdx] = EvaluatorUtils::scaleUpToZZ(uvals[i].imag(), plain.logp  + logQ);
		// cout << "plain[" << i << "]: " << plain.mx[idx] << " " << plain.mx[jdx] << endl;
	}
	delete[] uvals;
	// timeutils.stop("Encode");

	// for (int i = 0; i < n; i++)
	// {
	// 	cout << mvec[i] << endl;
	// }
	// cout << endl;
	
	// for (int i = 0, j = Nh, k = 0; k < n; i += gap, j += gap, k++) {
	// 	cout << "plain" << i << ": " << plain.mx[i] << " " << plain.mx[j] << endl;
	// }
	// cout << endl;

	// timeutils.start("Encrypt");
	cipher.logp = plain.logp;
	cipher.logq = plain.logq;
	cipher.n = plain.n;
	ZZ qQ = ring.qpows[plain.logq + logQ];

	ZZ* vx = new ZZ[N];
	ring.sampleZO(vx);

	Key* key = scheme.isSerialized ? SerializationUtils::readKey(scheme.serKeyMap.at(ENCRYPTION)) : scheme.keyMap.at(ENCRYPTION);

	long np = ceil((1 + logQQ + logN + 2)/(double)pbnd);
	// timeutils.start("Encrypt_multNTT");
	// ring.multNTT(cipher.ax, vx, key->rax, np, qQ);

	// ring.multNTT(cipher.bx, vx, key->rbx, np, qQ);
	uint64_t* rb = key->rax;
	ZZ* a = vx;
	ZZ& mod = qQ;
	// void RingMultiplier::multNTT(ZZ* x, ZZ* a, uint64_t* rb, long np, const ZZ& mod)
	uint64_t* ra = new uint64_t[np << logN]();
	uint64_t* rx = new uint64_t[np << logN]();
	NTL_EXEC_RANGE(np, first, last);
	cout << "first: " << first << " last: " << last << endl;
	for (long i = first; i < last; ++i) {
		uint64_t* rai = ra + (i << logN);
		uint64_t* rbi = rb + (i << logN);
		uint64_t* rxi = rx + (i << logN);
		uint64_t pi = ring.multiplier.pVec[i];
		uint64_t pri = ring.multiplier.prVec[i];
		_ntl_general_rem_one_struct* red_ss = ring.multiplier.red_ss_array[i];
		for (long n = 0; n < N; ++n) {
			rai[n] = _ntl_general_rem_one_struct_apply(a[n].rep, pi, red_ss);
		}
		ring.multiplier.NTT(rai, i);
		
		
		for (long n = 0; n < N; ++n) {
			ring.multiplier.mulModBarrett(rxi[n], rai[n], rbi[n], pi, pri);
		}
		for (long n = 0; n < N; n++)
		{
			cout << a[n] << " " << rai[n] << endl;
		}
		ring.multiplier.INTT(rxi, i);
	}
	NTL_EXEC_RANGE_END;

	ring.multiplier.reconstruct(cipher.ax, rx, np, mod);

	delete[] ra;
	delete[] rx;
	// timeutils.stop("Encrypt_multNTT");
	// delete[] vx;

	// ring.addAndEqual(cipher.bx, plain.mx, qQ);

	// ring.rightShiftAndEqual(cipher.ax, logQ);
	// ring.rightShiftAndEqual(cipher.bx, logQ);

	// ring.addGaussAndEqual(cipher.ax, qQ);
	// ring.addGaussAndEqual(cipher.bx, qQ);
	// timeutils.stop("Encrypt");

	// timeutils.start("Decrypt");
	// ZZ q = ring.qpows[cipher.logq];
	// plain.logp = cipher.logp;
	// plain.logq = cipher.logq;
	// plain.n = cipher.n;
	// np = ceil((1 + cipher.logq + logN + 2)/(double)pbnd);
	// ring.mult(plain.mx, cipher.ax, secretKey.sx, np, q);
	// ring.addAndEqual(plain.mx, cipher.bx, q);
	// timeutils.stop("Decrypt");

	// timeutils.start("Decode");
	// // complex<double>* dvec = scheme.decode(plain);
	// // plain.mx, res, plain.n, plain.logp, plain.logq
	// ZZ* mx = plain.mx; 
	// complex<double>* vals = new complex<double>[plain.n];
	// q = (ring.qpows[plain.logq]);
	// // cout << "q: " << q << endl;
	
	// gap = Nh / slots;
	// ZZ tmp;
	// for (long i = 0, idx = 0; i < slots; ++i, idx += gap) {
	// 	rem(tmp, mx[idx], q);
	// 	// cout << "tmp: " << tmp << endl;
	// 	if (NumBits(tmp) == logq) tmp -= q;
	// 	vals[i].real(EvaluatorUtils::scaleDownToReal(mx[idx], plain.logp));
		

	// 	rem(tmp, mx[idx + Nh], q);
	// 	if (NumBits(tmp) == logq) tmp -= q;
	// 	vals[i].imag(EvaluatorUtils::scaleDownToReal(mx[idx + Nh], plain.logp));

	// 	// cout << "val[" << i << "]: " << vals[i] << "," << tmp << endl;
	// }
	// ring.EMB(vals, slots);
	// timeutils.stop("Decode");

	// StringUtils::compare(mvec, vals, n, "val");

	// TestScheme::testEncrypt(logq, logp, logn);
	return 0;
}
