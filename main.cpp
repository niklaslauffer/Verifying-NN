#include <iostream>
#include <vector>
#include "build_network.h"
#include "DAG.h"


int main () {

    std::cout << "Building network...\n";
    Network<double> net = yml2network<double>("mnist_deep_nn/mnist_mlp_sigmoid.yml");
    std::cout << "Output:\n";
    std::vector<double> input = {-.5,+0.1};
    std::vector<double> output = net.eval(input);
    int out_size = output.size();
    for (int i = 0; i < out_size; i++) {
      std::cout << output[i] << std::endl;
    }
    return 0;
}

