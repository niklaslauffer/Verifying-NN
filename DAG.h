#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "aa_aaf.h"

template <typename T>
class Network {
  public:

    enum Transformation {
      none,
      sum,
      sigmoid_act,
      relu_act,
      tanh_act
    };

    class Node {
      public:
        Transformation type;
        double bias;
        std::vector<Node*> children; // pointers to the children
        std::vector<double> child_weights; // weights along the edges to the children
        std::vector<Node*> parents; // pointers to the parents
        std::vector<double> parent_weights; // weights along the edges to the children
        T value;

        void update_value();
    };

    void set_input_layer(std::vector<Node*> in);

    std::vector<T> eval(std::vector<T>& input);

  private:

    std::vector<Node*> input;

    // static T relu (T x);
    // static T sigmoid (T x);
    // static T tanh (T x);
};

double relu (double x) ;
double sigmoid (double x);
AAF relu (const AAF &val);
AAF sigmoid (const AAF &val);

#endif /* NETWORK_H */
