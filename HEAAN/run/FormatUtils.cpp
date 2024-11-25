#pragma once
#ifndef FORMATUTILS_CPP
#define FORMATUTILS_CPP

#include "FormatUtils.h"


using namespace std;

void print_rep_img(double* dec0, long w) {
    for (int i = 0; i < w * 4; i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i] * 1000)) / 1000;
        if ((i + 1) % w == 0) cout << endl;
    }
    cout << " ... " << endl;
    for (int i = w * w; i < w * (w + 4); i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i] * 1000)) / 1000;
        if ((i + 1) % w == 0) cout << endl;
    }
    cout << " ... " << endl;
    for (int i = 2 * w * w; i < 2 * w * (w + 2); i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i] * 1000)) / 1000;
        if ((i + 1) % w == 0) cout << endl;
    }
    cout << endl;
}

void print_rep(complex<double>* dec0, long n) {
    for (int i = 0; i < 10; i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i].real() * 1000)) / 1000 << "," << setw(6) << double(int(dec0[i].imag() * 1000)) / 1000 << "  ";
        if ((i + 1) % 5 == 0) cout << endl;
    }
    cout << " ... " << endl;
    for (int i = n/2 - 5; i < n/2 + 5; i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i].real() * 1000)) / 1000 << "," << setw(6) << double(int(dec0[i].imag() * 1000)) / 1000 << "  ";
        if ((i + 1) % 5 == 0) cout << endl;
    }
    cout << " ... " << endl;
    for (int i = n - 10; i < n; i++) {
        cout << setw(6) << setprecision(3) << double(int(dec0[i].real() * 1000)) / 1000 << "," << setw(6) << double(int(dec0[i].imag() * 1000)) / 1000 << "  ";
        if ((i + 1) % 5 == 0) cout << endl;
    }
    cout << endl;
}

void print_shape(complex<double>* mvec, long w, long c, std::string dir) {
    if (dir != "") {
        ofstream file(dir);
        if (!file.is_open()) {
            cout << "Cannot open file " << dir << endl;
            return;
        }
        for (int i = 0; i < c; i++) {
            for (int j = 0; j < w; j++) {
                for (int k = 0; k < w; k++) {
                    file << setw(10) << setprecision(4) << mvec[i * w * w + j * w + k].real() << " ";
                }
                file << endl;
            }
            file << endl;
        }
        file << endl;
        cout << "Saved to " << dir << endl;
        file.close();
    } else {
        for (int i = 0; i < c; i++) {
            for (int j = 0; j < w; j++) {
                for (int k = 0; k < w; k++) {
                    cout << mvec[i * w * w + j * w + k].real() << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }
}

void print_res_classification(complex<double>* mvec) {
    double max_val = -100;
    int max_idx = -1;
    for (int i = 0; i < 10; i++) {
        cout << setw(8) << mvec[i * 64].real() << " ";
        if (mvec[i * 64].real() > max_val) {
            max_val = mvec[i * 64].real();
            max_idx = i;
        }
    }
    cout << endl;
    cout << "Max value: " << max_val << " at index: " << max_idx << endl;
}

void readImage(string path, double*& image, int& w, int& h, int& c) {
    uint8_t* rgb_image = stbi_load(path.c_str(), &w, &h, &c, 3);
    for (int i = 0; i < w * h * c; i++) {
        image[i] = static_cast<double>(rgb_image[i]) / 255.0;
    }
    stbi_image_free(rgb_image);
}

#endif // !FORMATUTILS_CPP
