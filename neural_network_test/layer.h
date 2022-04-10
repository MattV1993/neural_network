#pragma once

#include "neuron.h"
#include "layer_spec.h"

namespace neural_network
{
	class layer
	{
	public:

		layer(const layer_spec& spec);

		void set_connections(layer& next_layer);
		void set_values(const std::vector<double>& values);

		void feed_forward();
		void calculate_gradiant(const std::vector<double>& targets);
		void calculate_gradiant();
		void update_weights(double learning_rate, double eta);

		const neuron_list& neurons() const;
		neuron_list& neurons();

	private:

		layer_spec spec;
		neuron_list neurons_;
	};
}