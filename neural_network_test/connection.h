#pragma once

namespace neural_network
{

	class neuron;

	class connection
	{
	public:

		connection(neuron& input_neuron, neuron& output_neuron, double weight);

		neuron& input_neuron() const;
		neuron& output_neuron() const;

		double weight() const;
		void set_weight(double value);

		double delta_weight() const;
		void set_delta_weight(double value);

	private:

		neuron& input_neuron_;
		neuron& output_neuron_;

		double weight_;
		double delta_weight_;
	};
};