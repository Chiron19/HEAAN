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
	long num_batches = 100;
	bool verbose = false;

	if (argc < 2 || argc > 5) {
		cerr << "Error: argument is required." << endl;
		cerr << "Usage: " << argv[0] << " <path_to_binary_file> [logq_for_dec] [num_batches] [verbose]" << endl;
		return 1;
	}
	else if (argc == 3) {
		logq_for_dec = atol(argv[2]);
	}
	else if (argc == 4) {
		logq_for_dec = atol(argv[2]);
		num_batches = atol(argv[3]);
	}
	else if (argc == 5) {
		logq_for_dec = atol(argv[2]);
		num_batches = atol(argv[3]);
		verbose = (atol(argv[4]) >= 1);
	}

	string filePath = argv[1];
	ifstream file(filePath, ios::binary);

	if (!file) {
		cerr << "Error: Unable to open file '" << filePath << "'" << endl;
		return 1;
	}

	// File opened successfully, you can now read from it
	// Read the file content
	file.seekg(0, ios::end);
	streamsize fileSize = file.tellg();
	file.seekg(0, ios::beg);

	// Calculate the number of float32 values
	long numFloatsTotal = fileSize / sizeof(float);

	// Make sure numFloatsTotal is a multiple of num_batches
	if (numFloatsTotal % num_batches != 0) {
		cerr << "Error: Number of float32 values is not a multiple of num_batches" << endl;
		return 1;
	}

	long numFloats = numFloatsTotal / num_batches;
	cout << "numFloats: " << numFloats << endl;
	cout << "num_batches: " << num_batches << endl;

	// Allocate memory for float and double arrays
	float** floatArray = new float*[num_batches];
	for (long i = 0; i < num_batches; i++) {
		floatArray[i] = new float[numFloats];
	}

	// Read float32 values from the file
	for (long i = 0; i < num_batches; i++) {
		file.read(reinterpret_cast<char*>(floatArray[i]), numFloats * sizeof(float));
	}

    file.close();

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

	if (verbose) {
		// Print the first few values for verification
		cout << "Quick View:" << endl;
		for (long i = 0; i < num_batches; i+= (num_batches / 100)) {
			cout << "Batch " << i << ": ";	
			for (long j = 0; j < min(10L, numFloats); ++j) {
				cout << setprecision(5) << floatArray[i][j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

    // Parameters //
    long logq = 300; ///< Ciphertext modulus (this value should be <= logQ in "scr/Params.h")
    long logp = 30; ///< Scaling Factor (larger logp will give you more accurate value)
    long logn = 15; ///< number of slot is 2^logn (this value should be < logN in "src/Params.h")
    long n = 1 << logn;
    long slots = n;
    long numThread = 8;

    // Construct and Generate Public Keys //
    srand(time(NULL));
    SetNumThreads(numThread);
    TimeUtils timeutils;
    Ring ring;

	// Load secret key from file if exists, otherwise generate a new one
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

	// Fake Decrypt //
    SecretKey secretKey_Fake(ring);
    for (long i = 0; i < N; i++) {
        secretKey_Fake.sx[i] = 0;
    }
	
	long num_batches_slice = 10;
	long slice = num_batches / num_batches_slice;

	cout << "Total Slices: " << slice << endl;

	for (long slice_i = 0; slice_i < slice; slice_i++) {
		cout << "Slice " << slice_i << endl;

		// Make Array of Complex //
		complex<double>** mvec1 = new complex<double>*[num_batches_slice];
		long halfLen = numFloats / 2;
		for (long i = 0; i < num_batches_slice; i++) {
			mvec1[i] = new complex<double>[n];
			// Fill real and imaginary parts
			for (long j = 0; j < halfLen; ++j) {
				mvec1[i][j] = complex<double>(static_cast<double>(floatArray[slice_i * num_batches_slice + i][j]), static_cast<double>(floatArray[slice_i * num_batches_slice + i][j + halfLen]));
			}
			
			// Fill remaining slots with zeros if necessary
			for (long j = halfLen; j < n; ++j) {
				mvec1[i][j] = complex<double>(0.0, 0.0);
			}
		}

		// Encrypt Complex Array //
		Ciphertext* cipher1 = new Ciphertext[num_batches_slice];
		for (long i = 0; i < num_batches_slice; i++) {
			if (verbose) cout << "Encrypting Batch " << i << endl;
			scheme.encrypt(cipher1[i], mvec1[i], n, logp, logq);
		}

		for (long i = 0; i < num_batches_slice; i++) {
			delete[] mvec1[i];
		}
		delete[] mvec1;
    
    	Plaintext* plain1 = new Plaintext[num_batches_slice];
		for (long i = 0; i < num_batches_slice; i++) {
			scheme.decryptMsg(plain1[i], secretKey_Fake, cipher1[i]);
		}

		for (long i = 0; i < num_batches_slice; i++) {
			cipher1[i].free();
		}
		delete[] cipher1;

		complex<double>** dec1 = new complex<double>*[num_batches_slice];
		for (long i = 0; i < num_batches_slice; i++) {
			dec1[i] = new complex<double>[n];
		}

		// expansion of ring.decode(plain1.mx, dec1, plain1.n, plain1.logp, plain1.logq);
		ZZ q = ring.qpows[logq];
		long gap = Nh / slots;
		ZZ tmp;
		for (long b = 0; b < num_batches_slice; b++) {	
			for (long i = 0, idx = 0; i < slots; ++i, idx += gap) {
				// printf("%ld %ld %ld\n", i, idx, idx + Nh);
				rem(tmp, plain1[b].mx[idx], q);
				if (NumBits(tmp) == logq) tmp -= q;
				dec1[b][i].real(EvaluatorUtils::scaleDownToReal(tmp, logq_for_dec));

				rem(tmp, plain1[b].mx[idx + Nh], q);
				if (NumBits(tmp) == logq) tmp -= q;
				dec1[b][i].imag(EvaluatorUtils::scaleDownToReal(tmp, logq_for_dec));
			}

			if (verbose) {
				cout << "Decrypted values for batch " << b + slice_i * num_batches_slice << ":" << endl;
				for (long i = 0; i < min(10L, numFloats); ++i) {
					cout << setprecision(5) << dec1[b][i].real() << " + 1j*" << dec1[b][i].imag() << " ";
				}
				cout << endl;
			}

			ring.EMB(dec1[b], slots);
		}

		for (long i = 0; i < num_batches_slice; i++) {
			plain1[i].free();
		}
		delete[] plain1;

		// Convert doubleArray_ to float32 array
		float** floatArray_ = new float*[num_batches_slice];
		for (long i = 0; i < num_batches_slice; i++) {
			floatArray_[i] = new float[numFloats];
		}
		
		for (long b = 0; b < num_batches_slice; b++) {
			for (long i = 0; i < halfLen; ++i) {
				floatArray_[b][i] = static_cast<float>(dec1[b][i].real());
				floatArray_[b][i + halfLen] = static_cast<float>(dec1[b][i].imag());
			}
		}

		for (long i = 0; i < num_batches_slice; i++) {
			delete[] dec1[i];
		}
		delete[] dec1;

		if (verbose) {
			// Print the first few values for verification
			cout << "Quick View:" << endl;
			for (long i = 0; i < num_batches_slice; i++) {
				cout << "Batch " << i + slice_i * num_batches_slice << ": ";	
				for (long j = 0; j < min(10L, numFloats); ++j) {
					cout << setprecision(5) << floatArray_[i][j] << " ";
				}
				cout << endl;
			}
			cout << endl;
		}


		cout << "Slice " << slice_i << " Writing to file: " << newFilepath << endl;
		
		// Open the file for writing in append mode
		ofstream outFile(newFilepath, ios::binary | ios::app);
		if (!outFile) {
			cerr << "Error: Unable to open file for writing: " << newFilepath << endl;
			return 1;
		}

		// Write the float array to the file 
		for (long i = 0; i < num_batches_slice; i++) {
			outFile.write(reinterpret_cast<char*>(floatArray_[i]), numFloats * sizeof(float));
		}

		// Close the file
		outFile.close();

		for (long i = 0; i < num_batches_slice; i++) {
			delete[] floatArray_[i];
		}
		delete[] floatArray_;

	}

	cout << "All decrypted data written to: " << newFilepath << endl;

	for (long i = 0; i < num_batches; i++) {
		delete[] floatArray[i];
	}
	delete[] floatArray;

	cout << "Success" << endl;

	return 0;
}
