#pragma once

#include "activation.h"

#include <vector>
#include <optional>
#include <functional>

namespace neural_network
{
	inline double rand();

	struct input_layer_spec
	{

	};

	struct output_layer_spec
	{
		output_layer_spec(
			size_t neuron_count,
			neural_network::activation_type activation_function,
			std::function<double()> inital_bias = rand);

		neural_network::activation_type activation_function;
		std::function<double()> inital_bias;

		size_t neuron_count;
	};

	struct layer_spec
	{
		layer_spec(
			size_t neuron_count,
			neural_network::activation_type activation_function,
			std::function<double()> inital_bias = rand,
			std::function<double()> initial_forward_connection_weight = rand);

		layer_spec(const output_layer_spec& output_spec);

		neural_network::activation_type activation_function;
		std::function<double()> inital_bias = rand;
		std::function<double()> initial_forward_connection_weight = rand;

		size_t neuron_count;
	};

	inline double rand()
	{
		return static_cast<double>(std::rand() / static_cast<double>(RAND_MAX));
	}
}