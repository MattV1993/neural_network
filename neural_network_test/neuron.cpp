#include "neuron.h"

neural_network::neuron::neuron(activation_type activation_function)
	: activation_function{ convert(std::move(activation_function)) },
	value_{ 0.0 }, bias_{ 0.0 }, gradiant_{ 0.0 }
{

}

void neural_network::neuron::feed_forward()
{
	double sum = 0.0;

	for (const auto& c : backward_connections_)
	{
		sum += c.input_neuron().value() * c.weight() + bias();
	}

	value_ = activation_function->f(sum);
}

double neural_network::neuron::value() const
{
	return value_;
}

void neural_network::neuron::set_value(double value)
{
	this->value_ = value;
}

double neural_network::neuron::bias() const
{
	return bias_;
}

double neural_network::neuron::gradiant() const
{
	return gradiant_;
}

void neural_network::neuron::calculate_gradiant(double delta)
{
	gradiant_ = delta * activation_function->derivative(value());
}

const std::vector<neural_network::connection>& neural_network::neuron::forward_connections() const
{
	return forward_connections_;
}

void neural_network::neuron::set_forward_connection(const connection& c)
{
	forward_connections_.push_back(c);
}

void neural_network::neuron::set_backward_connection(const connection& c)
{
	backward_connections_.push_back(c);
}

void neural_network::neuron::update_weights(double alpha, double eta)
{
	for (auto& c : backward_connections_)
	{
		neuron& prev_neuron = c.input_neuron();
		double current_delta_weight = c.delta_weight();
		double new_delta_weight = eta * prev_neuron.value() * gradiant() + alpha * current_delta_weight;

		c.set_delta_weight(new_delta_weight);
		c.set_weight(c.weight() + new_delta_weight);
	}
}