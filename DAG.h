#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "aa_aaf.h"
#include "aa_interval.h"

template <typename T>
class Network {
  public:

    enum Transformation {
      none,
      sum,
      sigmoid_act,
      relu_act,
      softmax,
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
    void apply_softmax(std::vector<Node*> next_layer, std::vector<Node*> prev_layer);

    // static T relu (T x);
    // static T sigmoid (T x);
    // static T tanh (T x);
};

double relu (double x) ;
double sigmoid (double x);
double tanh (double x);
double softmax (double x);
AAF relu (const AAF &val);
AAF sigmoid (const AAF &val);
AAF softmax (const AAF &val);
// AAF tanh ( AAF x);

#endif /* NETWORK_H */
