#include "tensor/tensor.hpp"
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// command :
// perf stat -e cycles,instructions,cache-references,cache-misses ./train

using namespace std;

#define DEFAULT_SIZE 2048

int size_of_mat;

void randomize(Tensor &t) {
  vector<float> &data = *t.get_data_pointer();
  for (float &x : data) {
    x = ((float)rand() / RAND_MAX) - 0.5f;
  }
}

int main(int argc, char *argv[]) {
  if (argc == 2)
    size_of_mat = stoi(argv[1]);
  else
    size_of_mat = DEFAULT_SIZE;

  Tensor t1{{size_of_mat, size_of_mat}};
  Tensor t2{{size_of_mat, size_of_mat}};
  randomize(t1);
  randomize(t2);

  cout << "Matmul begins" << endl;
  Tensor out = t1.matmul_2D(t2);
  cout << "Matmul ended" << endl;

  return 0;
}

/*
 1) buildin tensor matmul_2D

    234,948,103,408      cycles
    61,458,759,363      instructions  #    0.26  insn per cycle
    8,695,158,605      cache-references
    5,567,695,159      cache-misses  #   64.03% of all cache refs

      56.351237922 seconds time elapsed

      56.199030000 seconds user
       0.092971000 seconds sys

 */

// Implement :
// RSMprop

// loading functions of model, layers, and tensors are defined in
// helpers.hpp and Implementend in helpers.cpp
//
// If needed move them to appropriate files
