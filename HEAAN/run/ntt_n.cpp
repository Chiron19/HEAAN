#include "../src/HEAAN.h"

#include <NTL/BasicThreadPool.h>
#include <NTL/ZZ.h>
#include <string>
#include <iomanip>

using namespace std;
using namespace NTL;
using namespace heaan;

/**
  * NTT generator
  * Allow assigning N (power of 2) for NTT
  */

void NTT(uint64_t* a, long index, long logN, RingMultiplier ringMultiplier) {
	long t = (1 << logN);
	long logt1 = logN + 1;
	uint64_t p = ringMultiplier.pVec[index];
	uint64_t pInv = ringMultiplier.pInvVec[index];
	for (long m = 1; m < (1 << logN); m <<= 1) {
		t >>= 1;
		logt1 -= 1;
		for (long i = 0; i < m; i++) {
			long j1 = i << logt1;
			long j2 = j1 + t - 1;
			uint64_t W = ringMultiplier.scaledRootPows[index][m + i];
			for (long j = j1; j <= j2; j++) {
				ringMultiplier.butt(a[j], a[j+t], W, p, pInv);
			}
		}
	}
}

int main(int argc, char **argv) {
    ofstream cout("ntt_n.csv", ios::app);

	srand(time(NULL));
	SetNumThreads(1);
	TimeUtils timeutils;
	Ring ring;

    long p = rand() % 28, logN = 2, N = (1 << logN);
    uint64_t* a = new uint64_t[N]; // Dynamically allocate memory for the array

    for (int i = 0; i < N; i++) {
        a[i] = rand(); // Assign a random value to each entry
        cout << a[i] << ((i != N - 1) ? "," : ""); // Print array a into one line, separated by comma
    }
    cout << endl;

    NTT(a, p, logN, ring.multiplier);

    for (int i = 0; i < N; i++) {
        cout << a[i] << ((i != N - 1) ? "," : "");
    }
    cout << endl;

    for (int i = 0; i < N; i++)
    {
        cout << ring.multiplier.scaledRootPows[p][i] << ((i != N - 1) ? "," : "");
    }
    cout << endl;
    
	return 0;
}
