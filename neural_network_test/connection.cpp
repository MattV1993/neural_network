#include "connection.h"

neural_network::connection::connection(neuron& input_neuron, neuron& output_neuron, double weight)
	: input_neuron_{ input_neuron }, output_neuron_{ output_neuron },
	weight_{ weight }, delta_weight_{ 0 }
{
}

neural_network::neuron& neural_network::connection::input_neuron() const
{
	return input_neuron_;
}

neural_network::neuron& neural_network::connection::output_neuron() const
{
	return output_neuron_;
}

double neural_network::connection::weight() const
{
	return weight_;
}

void neural_network::connection::set_weight(double value)
{
	weight_ = value;
}

double neural_network::connection::delta_weight() const
{
	return delta_weight_;
}

void neural_network::connection::set_delta_weight(double value)
{
	delta_weight_ = value;
}
