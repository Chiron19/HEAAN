#include "../src/HEAAN.h"

#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <string>
#include <iomanip>

using namespace std;
using namespace NTL;
using namespace heaan;

void secretKeyFootprint(SecretKey secretKey) {
    string hashString(64, '0');
    for (long i = 0; i < N; i++) {
        long index = i % 64;
        if (secretKey.sx[i] == 1) {
            hashString[index] = (hashString[index] == '0') ? '1' : (hashString[index] == '1') ? '2' : '0';
        } else if (secretKey.sx[i] == -1) {
            hashString[index] = (hashString[index] == '0') ? '2' : (hashString[index] == '1') ? '0' : '1';
        }
    }
    cout << hashString << endl;
}

int main(int argc, char **argv) {
	long logq_for_dec = 300;
	if (argc == 3) {
		logq_for_dec = atol(argv[2]);
	}
	else {
		if (argc != 2) {
			cerr << "Error: argument is required." << endl;
			cerr << "Usage: " << argv[0] << " <path_to_binary_file> [logq_for_dec]" << endl;
			return 1;
		}
	}

	string filePath = argv[1];
	ifstream file(filePath, ios::binary);

	if (!file) {
		cerr << "Error: Unable to open file '" << filePath << "'" << endl;
		return 1;
	}

	// File opened successfully, you can now read from it
	// For example:
	// char buffer[1024];
	// file.read(buffer, sizeof(buffer));
	// Read the file content
	file.seekg(0, ios::end);
	streamsize fileSize = file.tellg();
	file.seekg(0, ios::beg);

	// Calculate the number of float32 values
	long numFloats = fileSize / sizeof(float);
	if (numFloats > N) {
		cerr << "Error: Number of float32 values exceeds N" << endl;
		return 1;
	}

	cout << "numFloats: " << numFloats << endl;

	// Allocate memory for float and double arrays
	float* floatArray = new float[numFloats];
	double* doubleArray = new double[numFloats];

	// Read float32 values from the file
	file.read(reinterpret_cast<char*>(floatArray), fileSize);

    file.close();

	// Convert float32 to double
	for (long i = 0; i < numFloats; ++i) {
		doubleArray[i] = static_cast<double>(floatArray[i]);
	}

    delete[] floatArray;

	// Print the first few values for verification
	cout << "First 20 values:" << endl;
	for (long i = 0; i < min(20L, numFloats); ++i) {
		cout << setprecision(10) << doubleArray[i] << " ";
	}
    cout << endl;

    // Parameters //
    long logq = 300; ///< Ciphertext modulus (this value should be <= logQ in "scr/Params.h")
    long logp = 30; ///< Scaling Factor (larger logp will give you more accurate value)
    long logn = 15; ///< number of slot is 2^logn (this value should be < logN in "src/Params.h")
    long n = 1 << logn;
    long slots = n;
    long numThread = 8;

    numFloats = min(numFloats, n * 2);

    // Construct and Generate Public Keys //
    srand(time(NULL));
    SetNumThreads(numThread);
    TimeUtils timeutils;
    Ring ring;
    SecretKey secretKey(ring);

    unsigned char* secretKeyArray =  new unsigned char[N];

    ifstream infile("secretKey.bin");
    if (infile.good()) {
        infile.close();
        fstream fin;
		fin.open("secretKey.bin", ios::binary|ios::in);
		fin.read(reinterpret_cast<char*>(secretKeyArray), N * sizeof(unsigned char));
		fin.close();
		for (long i = 0; i < N; ++i) {
			secretKey.sx[i] = ZZFromBytes(&secretKeyArray[i], 1);
		}
    } else {
		for (long i = 0; i < N; ++i) {
			BytesFromZZ(&secretKeyArray[i], secretKey.sx[i], 1);
		}
		fstream fout;
		fout.open("secretKey.bin", ios::binary|ios::out);
		fout.write(reinterpret_cast<const char*>(secretKeyArray), N * sizeof(unsigned char));
		fout.close();
		cout << "secretKey.bin not found, created new one" << endl;
    }

    secretKeyFootprint(secretKey);

	delete[] secretKeyArray;
	
    Scheme scheme(secretKey, ring);
    // scheme.addLeftRotKeys(secretKey); ///< When you need left rotation for the vectorized message
    // scheme.addRightRotKeys(secretKey); ///< When you need right rotation for the vectorized message

    // Make Array of Complex //
	complex<double>* mvec1 = new complex<double>[n];
	long halfLen = numFloats / 2;
	
	// Fill real and imaginary parts
	for (long i = 0; i < halfLen; ++i) {
		mvec1[i] = complex<double>(doubleArray[i], doubleArray[i + halfLen]);
	}
	
	// Fill remaining slots with zeros if necessary
	for (long i = halfLen; i < n; ++i) {
		mvec1[i] = complex<double>(0.0, 0.0);
	}

    delete[] doubleArray;
	
    // Encrypt Complex Array //
	Ciphertext cipher1;
	scheme.encrypt(cipher1, mvec1, n, logp, logq);

    delete[] mvec1;

    cout << "Encrypted: " << cipher1.n << " " << cipher1.logq << " " << cipher1.logp << endl;
    cout << cipher1.bx[0] << endl;

	Ciphertext cipher2;
	scheme.reScaleTo(cipher2, cipher1, logq);

    // Fake Decrypt //
    SecretKey secretKey_Fake(ring);
    for (long i = 0; i < N; i++) {
        secretKey_Fake.sx[i] = 0;
    }
    Plaintext plain1;
    scheme.decryptMsg(plain1, secretKey_Fake, cipher2);

    cout << "Decrypted: " << plain1.n << " " << plain1.logq << " " << plain1.logp << endl;
    cout << NumBits(plain1.mx[0]) << "bits" << endl;
    cout << plain1.mx[0] << endl;

    complex<double>* dec1 = new complex<double>[plain1.n];
	// this is expansion of ring.decode(plain1.mx, dec1, plain1.n, plain1.logp, plain1.logq);
    ZZ q = ring.qpows[plain1.logq];
	long gap = Nh / slots;
	ZZ tmp;
	for (long i = 0, idx = 0; i < slots; ++i, idx += gap) {
        // printf("%ld %ld %ld\n", i, idx, idx + Nh);
		rem(tmp, plain1.mx[idx], q);
        if (NumBits(tmp) == logq) tmp -= q;
        if (idx == 0) {
            cout << "tmp: " << NumBits(tmp) << " " << tmp << endl;
            cout << "logq: " << logq << endl;
            cout << "logp: " << logp << endl;
        }
		dec1[i].real(EvaluatorUtils::scaleDownToReal(tmp, logq_for_dec));

		rem(tmp, plain1.mx[idx + Nh], q);
		if (NumBits(tmp) == logq) tmp -= q;
        if (idx == 0) {
            cout << "tmp: " << NumBits(tmp) << " " << tmp << endl;
            cout << "logq: " << logq << endl;
            cout << "logp: " << logp << endl;
        }
		dec1[i].imag(EvaluatorUtils::scaleDownToReal(tmp, logq_for_dec));
	}
    for (long i = 0; i < min(10L, numFloats); ++i) {
		cout << setprecision(10) << dec1[i].real() << " + 1j*" << dec1[i].imag() << " ";
	}
    cout << endl;
	ring.EMB(dec1, slots);

	// Create a new double array to store the decrypted values
	double* doubleArray_ = new double[numFloats];

	// Extract real and imaginary parts from dec1
	for (long i = 0; i < halfLen; ++i) {
		doubleArray_[i] = dec1[i].real();
		doubleArray_[i + halfLen] = dec1[i].imag();
	}

    delete[] dec1;

	// Convert doubleArray_ to float32 array
	float* floatArray_ = new float[numFloats];
	
	for (long i = 0; i < numFloats; ++i) {
		floatArray_[i] = static_cast<float>(doubleArray_[i]);
	}

    delete[] doubleArray_;

	// Print the first few values for verification
	cout << "First 20 decrypted values:" << endl;
	for (long i = 0; i < min(20L, numFloats); ++i) {
		cout << setprecision(10) << floatArray_[i] << " ";
	}
    cout << endl;

	// Extract the directory path and filename
	string filepath = argv[1];
	size_t lastSlash = filepath.find_last_of("/\\");
	string directory = filepath.substr(0, lastSlash + 1);
	string filename = filepath.substr(lastSlash + 1);

	// Insert "_" before the file extension
	size_t lastDot = filename.find_last_of(".");
	string newFilename = filename.substr(0, lastDot) + "_" + filename.substr(lastDot);

	// Combine to create the new filepath
	string newFilepath = directory + newFilename;

	// Open the file for writing
	ofstream outFile(newFilepath, ios::binary);
	if (!outFile) {
		cerr << "Error: Unable to open file for writing: " << newFilepath << endl;
		return 1;
	}

	// Write the float array to the file
	outFile.write(reinterpret_cast<char*>(floatArray_), numFloats * sizeof(float));

	// Close the file
	outFile.close();

	cout << "Decrypted data written to: " << newFilepath << endl;
	delete[] floatArray_;

	return 0;
}
