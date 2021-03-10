#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <fstream>
#include <string>

#include <cstdlib>
#include <ctime>

#include "build_network.h"
#include "DAG.h"
#include "aa_aaf.h"
#include "aa_interval.h"

#include <map>
#include <iterator>

int main () {
    srand (static_cast <unsigned> (time(0)));
    Network<AAF> net = yml2network<AAF>("wildlife_model_sigmoid.yaml");

    std::vector<double> nn_input_0;
    fstream file0;
    file0.open("kaggle_wildlife/wildlife_examples/0.txt",ios::in); //open a file to perform read operation using file object
    if (file0.is_open()){   //checking whether the file is open
       string tp;
       while(getline(file0, tp)){ //read data from file object and put it into string.
         nn_input_0.push_back( stof(tp) );
       }
       file0.close(); //close the file object.
    }

    std::vector<double> nn_input_1;
    fstream file1;
    file1.open("kaggle_wildlife/wildlife_examples/1.txt",ios::in); //open a file to perform read operation using file object
    if (file1.is_open()){   //checking whether the file is open
       string tp;
       while(getline(file1, tp)){ //read data from file object and put it into string.
         nn_input_1.push_back( stof(tp) );
       }
       file1.close(); //close the file object.
    }

    std::vector<double> nn_input_2;
    fstream file2;
    file2.open("kaggle_wildlife/wildlife_examples/2.txt",ios::in); //open a file to perform read operation using file object
    if (file2.is_open()){   //checking whether the file is open
       string tp;
       while(getline(file2, tp)){ //read data from file object and put it into string.
         nn_input_2.push_back( stof(tp) );
       }
       file2.close(); //close the file object.
    }

    int value = 0;
    typedef std::pair<int, double> Key;
    std::map<Key,std::vector<AAF>> alreadyCompute;

    fstream file;
    file.open("uncertain_class.txt",ios::in); //open a file to perform read operation using file object
    ofstream file_out;
    file_out.open("uncertain_class_out.txt"); //open a file to perform read operation using file object
    if (file.is_open()){   //checking whether the file is open
       string tp;
       while(getline(file, tp)){ //read data from file object and put it into string.
         value++;
         std::stringstream ss(tp);
         std::string token;
         std::vector<double> res;
         while( std::getline(ss, token, ',')){
          res.push_back(stof(token));
         }
         std::cout << value << ',' << res[5] << std::endl;
         // std::cout << res[0] << ',' << res[5] << std::endl;
         Key currKey = Key(static_cast<int>(res[0]), res[5]);
         if (alreadyCompute.count(currKey) > 0){
          std::vector<AAF> nn_output = alreadyCompute[currKey];
          file_out << tp;
          for (std::vector<AAF>::iterator it = nn_output.begin() ; it != nn_output.end(); ++it){
            file_out << "," << it->getMin() << ";" << it->getMax() << "";
          }
          file_out << "\n";
          continue;
         }

         if (res[0] == 0){ // Object ID is actually 0
          std::vector<AAF> nn_input_pert;
          for (std::vector<double>::iterator it = nn_input_0.begin() ; it != nn_input_0.end(); ++it){
            double minVal = (*it-res[5]) < 0 ? 0.0 : (*it-res[5]);
            double maxVal = (*it+res[5]) > 1 ? 1.0 : (*it+res[5]);
            float r3 = minVal + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(maxVal-minVal)));
            if (r3 > *it){
              maxVal = r3;
            }
            if (r3 < *it){
              minVal = r3;
            }
            if (*it < 1e-5){
              minVal = 0.0;
              maxVal = 0.0;
            }
            nn_input_pert.push_back(AAF(AAInterval(minVal, maxVal)));
          }
          std::vector<AAF> nn_output = net.eval(nn_input_pert);
          alreadyCompute[currKey] = nn_output;
          file_out << tp;
          for (std::vector<AAF>::iterator it = nn_output.begin() ; it != nn_output.end(); ++it){
            std::cout << '[' << it->getMin() << "," << it->getMax() << "] , ";
            file_out << "," << it->getMin() << ";" << it->getMax() << "";
          }
          file_out << "\n";
          std::cout << std::endl;
         }

         if (res[0] == 1){ // Object ID is actually 1
          std::vector<AAF> nn_input_pert;
          for (std::vector<double>::iterator it = nn_input_1.begin() ; it != nn_input_1.end(); ++it){
            double minVal = (*it-res[5]) < 0 ? 0.0 : (*it-res[5]);
            double maxVal = (*it+res[5]) > 1 ? 1.0 : (*it+res[5]);
            float r3 = minVal + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(maxVal-minVal)));
            if (r3 > *it){
              maxVal = r3;
            }
            if (r3 < *it){
              minVal = r3;
            }
            if (*it < 1e-5){
              minVal = 0.0;
              maxVal = 0.0;
            }
            nn_input_pert.push_back(AAF(AAInterval(minVal, maxVal)));
          }
          std::vector<AAF> nn_output = net.eval(nn_input_pert);
          alreadyCompute[currKey] = nn_output;
          file_out << tp;
          for (std::vector<AAF>::iterator it = nn_output.begin() ; it != nn_output.end(); ++it){
            std::cout << '[' << it->getMin() << "," << it->getMax() << "] , ";
            file_out << "," << it->getMin() << ";" << it->getMax() << "";
          }
          file_out << "\n";
          std::cout << std::endl;
         }

         if (res[0] == 2){
          std::vector<AAF> nn_input_pert;
          for (std::vector<double>::iterator it = nn_input_2.begin() ; it != nn_input_2.end(); ++it){
            double minVal = (*it-res[5]) < 0 ? 0.0 : (*it-res[5]);
            double maxVal = (*it+res[5]) > 1 ? 1.0 : (*it+res[5]);
            float r3 = minVal + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(maxVal-minVal)));
            if (r3 > *it){
              maxVal = r3;
            }
            if (r3 < *it){
              minVal = r3;
            }
            if (*it < 1e-5){
              minVal = 0.0;
              maxVal = 0.0;
            }
            nn_input_pert.push_back(AAF(AAInterval(minVal, maxVal)));
          }
          std::vector<AAF> nn_output = net.eval(nn_input_pert);
          alreadyCompute[currKey] = nn_output;
          file_out << tp;
          for (std::vector<AAF>::iterator it = nn_output.begin() ; it != nn_output.end(); ++it){
            std::cout << '[' << it->getMin() << "," << it->getMax() << "] , ";
            file_out << "," << it->getMin() << ";" << it->getMax() << "";
          }
          file_out << "\n";
          std::cout << std::endl;
         }

       }
       // Add more if clauses if more object labels

       file.close(); //close the file object.
       file_out.close();
    }

    return 0;
}
