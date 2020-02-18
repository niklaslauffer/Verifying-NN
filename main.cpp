#include <iostream>
#include <vector>
#include "build_network.h"
#include "DAG.h"


int main () {

    Network net = yml2network("sig16x16.yml");
    std::cout << "Output:\n";
    std::vector<double> input = {-1,-20.5};
    std::vector<double> output = net.eval(input);
    int out_size = output.size();
    for (int i = 0; i < out_size; i++) {
      std::cout << output[i] << std::endl;
    }
    return 0;
}

