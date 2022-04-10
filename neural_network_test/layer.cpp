#include "layer.h"
#include "connection.h"

neural_network::layer::layer(const layer_spec& spec)
	: spec{ spec }
{
	for (uint32_t i = 0; i < spec.neuron_count; i++)
	{
		neurons_.emplace_back(new neuron{ spec.activation_function });
	}
}

void neural_network::layer::set_connections(layer& next_layer)
{
	for (auto& forward_neuron : next_layer.neurons())
	{
		for (auto& neuron : neurons())
		{
			connection c{ *neuron, *forward_neuron, spec.initial_forward_connection_weight() };
			forward_neuron->set_backward_connection(c);
			neuron->set_forward_connection(c);
		}
	}
}

void neural_network::layer::set_values(const std::vector<double>& values)
{
	for (size_t i = 0; i < values.size(); i++)
	{
		neurons()[i]->set_value(values[i]);
	}
}

void neural_network::layer::feed_forward()
{
	for (auto& neuron : neurons())
	{
		neuron->feed_forward();
	}
}

void neural_network::layer::calculate_gradiant(const std::vector<double>& targets)
{
	for (size_t i = 0; i < neurons().size(); i++)
	{
		double delta = targets[i] - neurons()[i]->value();
		neurons()[i]->calculate_gradiant(delta);
	}
}

void neural_network::layer::calculate_gradiant()
{
	for (const auto& n : neurons())
	{
		double weight_derivatives = 0.0;

		for (const auto& c : n->forward_connections())
		{
			weight_derivatives += c.weight() * c.output_neuron().gradiant();
		}

		n->calculate_gradiant(weight_derivatives);
	}
}

void neural_network::layer::update_weights(double learning_rate, double eta)
{
	for (const auto& n : neurons())
	{
		n->update_weights(learning_rate, eta);
	}
}

const neural_network::neuron_list& neural_network::layer::neurons() const
{
	return neurons_;
}

neural_network::neuron_list& neural_network::layer::neurons()
{
	return neurons_;
}
