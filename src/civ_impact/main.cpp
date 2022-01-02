//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S I M P L E   F U N C T I O N   R E G R E S S I O N   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

#include <omp.h>

// OpenNN includes

#include "opennn/opennn/opennn.h"

static std::string GetCWD()
{
  static const std::string APP_NAME{"civ_impact"};
  static const size_t      NAME_LENGTH = APP_NAME.size();
  std::string full_path{realpath("/proc/self/exe", NULL)};
  return full_path.substr(0, full_path.size() - ( NAME_LENGTH + 1));
}

using namespace OpenNN;
using namespace std;
using namespace Eigen;

int main(void)
{
    try
    {
        cout << "Computing Civilizational Impact." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set
        std::string datapath = GetCWD() + "/data/civ_impact_data.csv";
        DataSet data_set(datapath, ';', true);

        data_set.split_samples_random();

        const Index input_variables_number = data_set.get_input_variables_number();

        const Index target_variables_number = data_set.get_target_variables_number();

        Tensor<string, 1> scaling_inputs_methods(input_variables_number);

        Tensor<string, 1> scaling_targets_methods(target_variables_number);

        scaling_inputs_methods.setConstant("MinimumMaximum");

        scaling_targets_methods.setConstant("MinimumMaximum");

        const Tensor<Descriptives, 1> inputs_descriptives = data_set.scale_input_variables(scaling_inputs_methods);

        const Tensor<Descriptives, 1> targets_descriptives = data_set.scale_target_variables(scaling_inputs_methods);

        // Neural network

        Tensor<Index, 1> neural_network_architecture(3);

        neural_network_architecture.setValues({1, 3, 1});

        NeuralNetwork neural_network(NeuralNetwork::Approximation, neural_network_architecture);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::MEAN_SQUARED_ERROR);

        training_strategy.set_optimization_method(TrainingStrategy::ADAPTIVE_MOMENT_ESTIMATION);

        training_strategy.get_loss_index_pointer()->set_regularization_method("NO_REGULARIZATION");

        training_strategy.get_adaptive_moment_estimation_pointer()->set_display_period(100);

        training_strategy.perform_training();

        data_set.unscale_input_variables(scaling_inputs_methods, inputs_descriptives);

        data_set.unscale_target_variables(scaling_targets_methods, targets_descriptives);

        neural_network.save_expression_python("simple_function_regresion.py");

        cout<<"Bye Simple Function Regression"<<endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques SL
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
