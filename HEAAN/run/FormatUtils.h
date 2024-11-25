#pragma once
#ifndef FORMATUTILS_H
#define FORMATUTILS_H

#include "../src/HEAAN.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <set>
#include <string>
#include <string_view>
#include <iterator>
#include <iomanip>
#include <iostream>
#include <complex>
#include <fstream>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif // !STB_IMAGE_IMPLEMENTATION

using namespace std;

void print_rep_img(double* dec0, long w);
void print_rep(complex<double>* dec0, long n);
void print_shape(complex<double>* mvec, long w, long c, std::string dir="");
void print_res_classification(complex<double>* mvec);
void readImage(string path, double*& image, int& w, int& h, int& c);

#endif // !FORMATUTILS_H