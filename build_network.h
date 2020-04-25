#include "DAG.h"
#include <string>
#include <iostream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#ifndef YML2NETWORK_H
#define YML2NETWORK_H

template <typename T>
Network<T> yml2network(std::string filename) {
  YAML::Node config = YAML::LoadFile(filename);

  // write now the topology is hard coded
  std::vector<int> topology {2,16,16,1};
  int input_size = topology[0];

  Network<T> net = Network<T>();

  // get parameters from the YAML file
  YAML::Node weights =  config["weights"];
  YAML::Node offsets =  config["offsets"];
  YAML::Node activations =  config["activations"];

  // construct the input layer
  std::vector<typename Network<T>::Node*> current_layer;
  current_layer.resize(input_size);
  // construct the input nodes
  for (int i = 0; i < input_size; i++) {
    current_layer[i] = new typename Network<T>::Node();
  }
  net.set_input_layer(current_layer);

  std::vector<typename Network<T>::Node*> next_layer_activ;

  // construct each layer of the network
  for (std::size_t i=1; i < topology.size(); i++) {
    // get info for this layer
    YAML::Node layer_weights = weights[i];
    YAML::Node layer_offsets = offsets[i];
    std::size_t next_layer_size = topology[i];


    // populate the next layer
    std::vector<typename Network<T>::Node*> next_layer;
    next_layer.resize(next_layer_size);
    for (std::size_t k = 0; k < next_layer_size; k++) {
      next_layer[k] = new typename Network<T>::Node();
    }
    // set biases of the next layer
    for (std::size_t k=0; k < next_layer_size; k++) {
      next_layer[k]->bias = layer_offsets[k].as<double>();
    }
    // set pointers and weights from current_layer to next_layer
    for (std::size_t j=0; j < current_layer.size(); j++) {
      for (std::size_t k=0; k < next_layer_size; k++) {
        // push_back pointer connections
        current_layer[j]->children.push_back(next_layer[k]);
        next_layer[k]->parents.push_back(current_layer[j]);
        // push_back weights
        current_layer[j]->child_weights.push_back(layer_weights[k][j].as<double>());
        next_layer[k]->parent_weights.push_back(layer_weights[k][j].as<double>());
        // set layer type
        next_layer[k]->type = Network<T>::sum;
      }
    }

    std::cout << current_layer.size() << std::endl;

    /* this next section sets up the activation layer */

    // populate the activation layer
    next_layer_activ.clear();
    next_layer_activ.resize(next_layer_size);
    for (std::size_t k = 0; k < next_layer_size; k++) {
      next_layer_activ[k] = new typename Network<T>::Node();
    }
    // add connections between next_layer and next_layer_activ
    for (std::size_t k=0; k < next_layer_size; k++) {
      // add pointer connections
      next_layer[k]->children.push_back(next_layer_activ[k]);
      next_layer_activ[k]->parents.push_back(next_layer[k]);
      // add trivial weights and bias
      next_layer[k]->child_weights.push_back(1);
      next_layer_activ[k]->parent_weights.push_back(1);
      next_layer_activ[k]->bias = 0;
      // set proper activation type
      std::string activ_type = activations[i].as<std::string>();
      if (activ_type.compare("Sigmoid") == 0) {
        next_layer_activ[k]->type = Network<T>::sigmoid_act;
      }
      else if (activ_type.compare("Tanh") == 0) {
        next_layer_activ[k]->type = Network<T>::tanh_act;
      }
      else {
        std::cerr << "Unsupported activation function type";
      }
    }
    // move to the next layer
    /* current_layer = std::move(next_layer_activ); */
    current_layer = std::move(next_layer_activ);
  }

  return net;
}

#endif /* NETWORK_H */
