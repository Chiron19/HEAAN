# Quick Start
The repository inherits the original features of HEAAN while adding children class to extend its utility.

## Prerequisites
- [`GMP`](https://gmplib.org)
It is highly recommended to accelerate `NTL` components.

- [`NTL`](http://www.shoup.net/ntl/)
Enabling number theory computations in C++.

## Build
1. `cd HEAAN/lib` to enter the building directory.
2. `make` to run the makefile.

## Test
1. `cd ../run` to enter the testing directory.
2. `make test` to compile the `test.cpp` into `TestHEAAN` executable.
3. `./TestHEAAN Encrypt` to run the executable.
4. `make clean` to remove the executable.

## Test Demo Model
1. Ensure you are under `run` directory.
2. Execute the `run.sh`, it is shell script to automatically compile the `new.cpp` into `new` executable and directly run it.
3. The secret key `secretKey.bin` is generated and serialized under the current directory by default. You may specify to another path (see code implementation in `new.cpp`).
4. The public key set and ciphertext checkpoints are serialized and saved under `serkey` and `cipher` directories. You may also specify to another paths.
5. The result is printed on the console with 10 float numbers, indicating the prediction of each class, and the classification result is the index where the max value located.

Example:
```c++
1.84347e-06 2.0560598e-06 4.6995515e-07 1.218864e-06 -1.2269359e-06 6.6855303e-07 1.1520117e-06 -3.0805123e-06 8.7770081e-07 -1.9588662e-07 
Max value: 2.0560598e-06 at index: 1
```

## Test Layer Precision
1. Compile the `test_layer.cpp` with same procedure above, modify the script file to accommodate your needs.

## Python for Auxiliary
1. Under the `Python` directory, there are some auxiliary programmes to play around CIFAR-10 datasets. All programmes are self-contained Jupyter notebooks, import the corresponding library and feel free to play with them. I suggest to start with `heaan_demo.ipynb`.

# HEAAN
HEAAN is software library that implements homomorphic encryption (HE) that supports fixed point arithmetics.
This library supports approximate operations between rational numbers.
The approximate error depends on some parameters and almost same with floating point operation errors.
The scheme in this library is on the paper "Homomorphic Encryption for Arithmetic of Approximate Numbers" (https://eprint.iacr.org/2016/421.pdf).

## Notice: This Repository is No Longer Maintained

Please note that this repository is no longer actively maintained. Issues and pull requests may not receive responses, and updates will not be provided.

For those seeking alternatives and more up-to-date libraries, we recommend checking out the library currently managed by [CryptoLab Inc.](https://www.cryptolab.co.kr) under the name "HEaaN." You can find more information and resources on their official website at [HEaaN.it](https://heaan.it).

Thank you for your understanding.

## Notice
In "Params.h", 'pbnd' value is 59.0 by default.
If you are using NTL with "NTL_ENABLE_AVX_FFT=on", This option reduces that small-prime size bound from 60 bits to 50 bits (see https://www.shoup.net/ntl/doc/tour-changes.html).
For this reason, you need to change the setting to 49.0.

### Notification for Decryption Algorithm
For the application of our library, you have to make sure that the decrypted value is not revealed other than secret key owner in the scenario. For details, see [here](SECURITY.md).

## Version
v1.0 Implementation of Original HEAAN scheme

V1.1 Implementation of Original HEAAN scheme with bootstrapping (https://eprint.iacr.org/2018/153.pdf)

V2.1 Faster Implementation of HEAAN scheme

## Dependency
This library is written by c++ and using NTL library (http://www.shoup.net/ntl/).

## How to use this library?
### 1. Build a static library/Running test functions
You can make a static library by typing "make all" in the /lib directory. After successful compilation you can find a static library libHEAAN.a in the /lib directory.

After you build libHEAAN.a, you can run a test program in the /run directory. In run.cpp, you need uncomment tests you need and type "make" in the /run directory. This command will run exe file "HEAAN".

We checked the program was working well on Ubuntu 16.04.2 LTS. You need to install NTL (with GMP), pThread, libraries.

## License
Copyright (c) by CryptoLab inc.
This program is licensed under a
Creative Commons Attribution-NonCommercial 3.0 Unported License.
You should have received a copy of the license along with this
work.  If not, see <http://creativecommons.org/licenses/by-nc/3.0/>.

## Test
In /test folder, we have test.cpp.
You can compile this code using "make".
After that, ./TestHEAAN TEST_NAME will test HEAAN library.
TEST_NAME can be one of followings: Encrypt, EncryptSingle, Add, Mult, iMult, RotateFast, Conjugate

## Example
```c++
#include "HEAAN.h"

using namespace std;
using namespace NTL;

int main() {
  /*
  * Basic Parameters are in src/Params.h
  * If you want to use another parameter, you need to change src/Params.h file and re-complie this library.
  */

  // Parameters //
  long logq = 300; ///< Ciphertext modulus (this value should be <= logQ in "scr/Params.h")
  long logp = 30; ///< Scaling Factor (larger logp will give you more accurate value)
  long logn = 10; ///< number of slot is 1024 (this value should be < logN in "src/Params.h")
  long n = 1 << logn;
  long slots = n;
  long numThread = 8;
	
  // Construct and Generate Public Keys //
  srand(time(NULL));
  SetNumThreads(numThread);
  TimeUtils timeutils;
  Ring ring;
  SecretKey secretKey(ring);
  Scheme scheme(secretKey, ring);
  scheme.addLeftRotKeys(secretKey); ///< When you need left rotation for the vectorized message
  scheme.addRightRotKeys(secretKey); ///< When you need right rotation for the vectorized message
  
  // Make Random Array of Complex //
  complex<double>* mvec1 = EvaluatorUtils::randomComplexArray(slots);
  complex<double>* mvec2 = EvaluatorUtils::randomComplexArray(slots);
  
  // Encrypt Two Array of Complex //
  Ciphertext cipher1;
  scheme.encrypt(cipher1, mvec1, n, logp, logq);
  Ciphertext cipher2;
  scheme.encrypt(cipher2, mvec2, n, logp, logq);
  
  // Addition //
  Ciphertext cipherAdd;
  scheme.add(cipherAdd, cipher1, cipher2);
  
  // Multiplication And Rescale //
  Ciphertext cipherMult;
  scheme.mult(cipherMult, cipher1, cipher2);
  Ciphertext cipherMultAfterReScale;
  scheme.reScaleBy(cipherMultAfterReScale, cipherMult, logp);
  
  // Rotation //
  long idx = 1;
  Ciphertext cipherRot;
  scheme.leftRotateFast(cipherRot, cipher1, idx);
  
  // Decrypt //
  complex<double>* dvec1 = scheme.decrypt(secretKey, cipher1);
  complex<double>* dvec2 = scheme.decrypt(secretKey, cipher2);
  
  return 0;

}
  
```
