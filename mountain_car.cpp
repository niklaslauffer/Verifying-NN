#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>

#include "build_network.h"
#include "DAG.h"

int main () {

    double low_bound = -.4;
    double high_bound = -.6;
    // get a random number from [.4, .6]
    srand(time(NULL));
    double init_pos = low_bound + rand() / (RAND_MAX / (high_bound - low_bound));
    std::cout << init_pos << std::endl;
    double goal_pos = .45;
    int step_limit = 116;

    Network net = yml2network("sig16x16.yml");
    double p = init_pos; // position
    double p_next;
    double v = 0; // velocity
    double v_next; 
    double reward = 0;
    double reward_next;
    double u; // control

    int reached_goal = 0;

    for (int i = 0; i < step_limit; i++) {

      std::cout << "Step: " << i << std::endl;
      std::cout << "Position: " << p << std::endl;
      std::cout << "Reward: " << reward << std::endl;

      std::vector<double> nn_input = {p, v};
      std::vector<double> nn_output = net.eval(nn_input);

      u = nn_output[0];
      std::cout << "Control: " << u << std::endl;

      if (reached_goal) {
        std::cout << std::endl << "Success!" << std::endl;
        break;
      }

      p_next = p + v;
      v_next = v + 0.0015 * u - 0.0025 * cos(3 * p);

      reward_next = reward - 0.1*pow(u,2);
      if (!reached_goal & p > goal_pos) {
          reward_next += 100;
          reached_goal = 1;
      }

      if (p_next < -1.2) 
        p = -1.2;
      else if (p_next > .6)
        p = .6;
      else
        p = p_next;

      if (v_next < -0.07)
        v = -0.07;
      else if (v_next > 0.07)
        v = 0.07;
      else 
        v = v_next;

      reward = reward_next;

      std::cout << std::endl;

    }

    return 0;
}

