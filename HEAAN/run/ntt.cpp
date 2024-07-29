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
    ofstream cout("ntt_data.csv", ios::app);

	srand(time(NULL));
	SetNumThreads(1);
	TimeUtils timeutils;
	Ring ring;

    long np = ceil((1 + logQQ + logN + 2)/(double)pbnd);

    uint64_t* a = new uint64_t[N]; // Dynamically allocate memory for the array

    for (int i = 0; i < N; i++) {
        a[i] = rand(); // Assign a random value to each entry
        cout << a[i] << ((i != N - 1) ? "," : ""); // Print array a into one line, separated by comma
    }
    cout << endl;

    long p = rand() % np;

    ring.multiplier.NTT(a, p);

    for (int i = 0; i < N; i++) {
        cout << a[i] << ((i != N - 1) ? "," : "");
    }
    cout << endl;

    cout << ring.multiplier.pVec[p];

    for (int i = 0; i < N - 1; i++)
    {
        cout << ",0";
    }
    cout << endl;
    

	return 0;
}
