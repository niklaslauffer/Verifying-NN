#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

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
        double value;

        void update_value();
    };

    void set_input_layer(std::vector<Node*> in);

    std::vector<double> eval(std::vector<double>& input);

  private:

    std::vector<Node*> input;

    static double relu (double x);

    static double sigmoid (double x);
};

#endif /* NETWORK_H */
