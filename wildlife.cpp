#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <time.h>
#include <fstream>
#include <string>

#include "build_network.h"
#include "DAG.h"
#include "aa_aaf.h"

int main () {

    /* Network<double> net = yml2network<double>("mnist_deep_nn/mnist_mlp_sigmoid.yml"); */
    Network<double> net = yml2network<double>("wildlife_model_sigmoid.yaml");

    std::cout << "Finished building network" << std::endl;

    std::vector<double> nn_input;
    fstream file;
    file.open("kaggle_wildlife/wildlife_examples/2.txt",ios::in); //open a file to perform read operation using file object
    if (file.is_open()){   //checking whether the file is open
       string tp;
       while(getline(file, tp)){ //read data from file object and put it into string.
         nn_input.push_back( stof(tp) );
       }
       file.close(); //close the file object.
    }

    /* for (size_t i = 0; i < nn_input.size(); i++) { */ 
    /*   std::cout << i << ") " << nn_input[i] << "\n"; */
    /* } */

    std::cout << "Finished reading input" << std::endl;

    std::vector<double> nn_output = net.eval(nn_input);

    /* ofstream outfile; */
    /* outfile.open("mnist_example3_softmax.txt"); */

    // Print output
    /* for (size_t i = 0; i < nn_output.size(); i++) { */ 
    /*   outfile << i << ") " << nn_output[i] << "\n"; */
    /* } */
    for (size_t i = 0; i < nn_output.size(); i++) { 
      std::cout << i << ") " << nn_output[i] << "\n";
    }

    return 0;
}
