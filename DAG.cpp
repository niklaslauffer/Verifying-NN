#include "DAG.h"

#include <vector>
#include <iostream>

#include <assert.h>
#include <math.h>

template <typename T>
void Network<T>::Node::update_value() {
  // check if we're at a root node
  if (this->parents.size() == 0) {
    std::cerr << "Shouldn't call update_value() on an input node!";
  }
  int parent_size = this->parents.size();
  T total = 0;
  switch (this->type) {
    case sum:
      for (int i = 0; i < parent_size; i ++) {
        total += this->parents[i]->value * this->parent_weights[i];
      }
      this->value = total + this->bias;
      /* std::cout << "Sum Out: " << this->value << std::endl; */
      break;
    case sigmoid_act:
      /* std::cout << "Sig In: " << this->parents[0]->value << std::endl; */
      this->value = sigmoid(this->parents[0]->value) + this->bias;
      assert(this->bias == 0);
      /* std::cout << "Sig Out: " << this->value << std::endl; */
      break;
    case relu_act:
      /* std::cout << "relu In: " << this->parents[0]->value << std::endl; */
      this->value = relu(this->parents[0]->value) + this->bias;
      assert(this->bias == 0);
      /* std::cout << "relu Out: " << this->value << std::endl; */
      break;
    case tanh_act:
      /* std::cout << "tanh In: " << this->parents[0]->value << std::endl; */
      this->value = tanh(this->parents[0]->value) + this->bias;
      assert(this->bias == 0);
      /* std::cout << "tanh Out: " << this->value << std::endl; */
      break;
    default:
      std::cerr << "Unknown node type" << std::endl;
  }
}

// /template <typename T>
// T Network<T>::relu (T x) {
//   if (x <= 0)
//     return 0;
//   else
//     return x;
// }

double relu (double x) {
  if (x <= 0)
    return 0;
  else
    return x;
}

// template <typename T>
// T Network<T>::sigmoid (T x) {
//   /* return x / (1 + abs(x)); */
//   return 1 / ( 1 + exp(-x));
// }

double sigmoid(double x){
  return 1 / ( 1 + exp(-x));
}

// template <typename T>
// T Network<T>::tanh (T x) {
//   return std::tanh(x);
// }

/************************************************************
 * Method:        relu
 ************************************************************/
AAF relu (const AAF &val)
{
  if (val <= 0)
    return AAF(0.0);
  return AAInterval(0, val.getMax());
}

/************************************************************
 * Method:        sigmoid
 ************************************************************/
AAF sigmoid (const AAF &val)
{
  return 1.0 / (1 + exp(-val));
}

template <typename T>
void Network<T>::set_input_layer(std::vector<Network::Node*> in) {
  int input_size = in.size();
  input.resize(input_size);
  for (int i = 0; i < input_size; i++) {
    this->input[i] = in[i];
  }
}

template <typename T>
std::vector<T> Network<T>::eval(std::vector<T>& input) {

  // populate the input layer with values
  int size = this->input.size();
  for (int i = 0; i < size; i++) {
    this->input[i]->value = input[i];
  }

  std::vector<Network::Node*> next_layer;
  std::vector<Network::Node*> prev_layer;
  int input_size = this->input.size();
  prev_layer.resize(input_size);


  for (int i = 0; i < input_size; i++) {
    prev_layer[i] = this->input[i];
  }
  while (prev_layer[0]->children.size() != 0) {
    /* std::cout << "Updating layer of type: " << prev_layer[0]->children[0]->type << std::endl; */
    if (prev_layer[0]->children[0]->type == Network::sum) {
      /* next_layer = std::move(prev_layer[0]->children); */
      next_layer = prev_layer[0]->children;
      int next_size = next_layer.size();
      /* std::cout << "next layer size: " << next_size << std::endl; */
      for (int i = 0; i < next_size; i++) {
        next_layer[i]->update_value();
      }
    }
    else { // otherwise it's an activation layer
      int next_size = prev_layer.size();
      /* std::cout << "next layer size: " << next_size << std::endl; */
      next_layer.resize(next_size);
      for (int i = 0; i < next_size; i++) {
        next_layer[i] = prev_layer[i]->children[0];
        assert(prev_layer[i]->children.size() == 1);
      }
      for (int i = 0; i < next_size; i++) {
        next_layer[i]->update_value();
        /* std::cout << "Input: " << prev_layer[i]->value << std::endl; */
        /* std::cout << "Output: " << next_layer[i]->value << std::endl; */
      }

    }
    /* prev_layer = std::move(next_layer); */
    prev_layer = next_layer;
  }
  std::vector<T> return_vec;
  return_vec.resize(prev_layer.size());
  for (std::size_t i = 0; i < prev_layer.size(); i++) {
    return_vec[i] = prev_layer[i]->value;
  }
  return return_vec;
}

template class Network<double>;
template class Network<AAF>;
