#include "layer_spec.h"

neural_network::layer_spec::layer_spec(
	size_t neuron_count,
	neural_network::activation_type activation_function,
	std::function<double()> inital_bias,
	std::function<double()> initial_forward_connection_weight)
	: neuron_count{ neuron_count },
	activation_function{ activation_function },
	inital_bias{ inital_bias },
	initial_forward_connection_weight{ initial_forward_connection_weight }
{

}

neural_network::layer_spec::layer_spec(const output_layer_spec& output_spec)
	: neuron_count{ output_spec.neuron_count },
	activation_function{ output_spec.activation_function },
	inital_bias{ output_spec.inital_bias },
	initial_forward_connection_weight{ nullptr }
{

}

neural_network::output_layer_spec::output_layer_spec(size_t neuron_count,
	neural_network::activation_type activation_function,
	std::function<double()> inital_bias)
	: neuron_count{ neuron_count },
	activation_function{ activation_function },
	inital_bias{ inital_bias }
{

}