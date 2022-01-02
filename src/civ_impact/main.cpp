#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <omp.h>

#include "opennn/opennn/opennn.h"

static std::string GetCWD()
{
  static const std::string APP_NAME{"civ_impact"};
  static const size_t      NAME_LENGTH = APP_NAME.size();
  std::string full_path{realpath("/proc/self/exe", NULL)};
  return full_path.substr(0, full_path.size() - ( NAME_LENGTH + 1));
}

using namespace OpenNN;
using namespace Eigen;

static void PrintColumnInfo(const Tensor<DataSet::Column, 1>& columns)
{
  for (Index i = 0; i < columns.size(); i++)
  {
    std::cout << "Column " << i << ": "         << std::endl;
    std::cout << "   Name: " << columns(i).name << std::endl;

    if (columns(i).column_use == OpenNN::DataSet::VariableUse::Input)
      std::cout << "   Use: input" << std::endl;
    else
    if (columns(i).column_use == OpenNN::DataSet::VariableUse::Target)
      std::cout << "   Use: target" << std::endl;
    else
    if (columns(i).column_use == OpenNN::DataSet::VariableUse::UnusedVariable)
      std::cout << "   Use: unused" << std::endl;
    if (columns(i).type == OpenNN::DataSet::ColumnType::Categorical)
      std::cout << "   Categories: " << columns(i).categories << std::endl;
    std::cout << std::endl;
  }
}

static void PrintDataset(const DataSet& data_set)
{
  std::cout << "Input variables number: "  << data_set.get_target_variables_number() << std::endl;
  std::cout << "Target variables number: " << data_set.get_target_variables_number() << std::endl;
}

int main(int argc, char** argv)
{
  try
  {
    std::cout << "Computing Civilizational Impact." << std::endl;

    std::string datapath = GetCWD() + "/data/civ_impact_data.csv";
    DataSet     data_set(datapath, ',', true);

    data_set.set_input();
    data_set.set_column_use(0, DataSet::VariableUse::Target);

    const Index                      input_variables_number  = data_set.get_input_variables_number();
    const Index                      target_variables_number = data_set.get_target_variables_number();
    const Tensor<DataSet::Column, 1> columns                 = data_set.get_columns();

    PrintColumnInfo(columns);
    PrintDataset(data_set);

    // Neural network
    NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, {input_variables_number, 50, target_variables_number});

    PerceptronLayer* perceptron_layer_pointer = neural_network.get_first_perceptron_layer_pointer();
    perceptron_layer_pointer->set_activation_function("RectifiedLinear");

    Tensor<Layer*, 1> layers_pointers = neural_network.get_trainable_layers_pointers();

    for (Index i = 0; i < layers_pointers.size(); i++)
    {
      std::cout << "Layer "    << i << ": "                             << std::endl;
      std::cout << "   Type: " << layers_pointers(i)->get_type_string() << std::endl;

      if (layers_pointers(i)->get_type_string() == "Perceptron")
        std::cout << "   Activation: " << static_cast<PerceptronLayer*>(layers_pointers(i))->write_activation_function() << std::endl;
      if (layers_pointers(i)->get_type_string() == "Probabilistic")
        std::cout << "   Activation: " << static_cast<ProbabilisticLayer*>(layers_pointers(i))->write_activation_function() << std::endl;
    }

    // Training strategy

    TrainingStrategy training_strategy(&neural_network, &data_set);

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
    training_strategy.set_display_period(100);
    training_strategy.set_maximum_epochs_number(1000);

    training_strategy.perform_training();

    // Testing analysis
    const TestingAnalysis  testing_analysis(&neural_network, &data_set);
    const Tensor<Index, 2> confusion                     = testing_analysis.calculate_confusion();
    const Tensor<type, 1>  multiple_classification_tests = testing_analysis.calculate_multiple_classification_tests();

    std::cout << "Confusion matrix: "                                              << std::endl;
    std::cout << confusion                                                         << std::endl;
    std::cout << "Accuracy: " << multiple_classification_tests(0)*type(100) << "%" << std::endl;
    std::cout << "Error: "    << multiple_classification_tests(1)*type(100) << "%" << std::endl;

    return 0;
  }
  catch(const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
