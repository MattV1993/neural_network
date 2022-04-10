#include "network.h"

#include <iostream>
#include <cstdint>
#include <string>

// TODO: 
// Check error is correct - only works if more than 1 output neuron
// Add bias corectly. Not in hidden layer? Should it be a seperate neuron? Added to layer not individual neurons?
// Should neurons have an input value? Cmment from video.
// Regularisation
// Dropout
// Early stopping
// Different optimizer -gradient decent, adam
// Different loss
// Ways to sort data

void log_cb(const neural_network::network_log& log)
{
	std::cout << "Pass: " << log.pass << '\n';
	std::cout << "Inputs: ";

	for (double input : log.inputs)
	{
		std::cout << std::to_string(input) + " ";
	}

	std::cout << "\n";
	std::cout << "Outputs: ";

	for (double output : log.outputs)
	{
		std::cout << std::to_string(output) + " ";
	}

	std::cout << "\n";
	std::cout << "Targets: ";

	for (double target : log.targets)
	{
		std::cout << std::to_string(target) + " ";
	}

	std::cout << "\n";
	std::cout << "Recent average error: " << log.recent_average_error << '\n';
	std::cout << '\n';
}

int main()
{
	neural_network::network net
	{
		{
			{
				{
					2, neural_network::activation_type::tanh
				},
				{
					2, neural_network::activation_type::tanh
				}
			},
			{
				1, neural_network::activation_type::tanh
			},
			log_cb,
			0.8,
			0.15
		}
	};

	auto results = net.results();

	return 0;
}