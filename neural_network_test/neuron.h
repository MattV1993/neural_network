#pragma once

#include "activation.h"
#include "connection.h"

#include <vector>
#include <memory>

namespace neural_network
{
	class neuron
	{
	public:

		neuron(activation_type activation_function);

		neuron(const neuron& other) = delete;
		neuron(neuron&& other) = default;
		~neuron() = default;

		neuron& operator=(const neuron& other) = delete;
		neuron& operator=(neuron&& other) = default;

		// Calculates sum of previously layers and passes result through activation function
		void feed_forward();

		double value() const;
		void set_value(double value);

		double bias() const;

		double gradiant() const;
		void calculate_gradiant(double delta);

		const std::vector<connection>& forward_connections() const;

		void set_forward_connection(const connection& c);
		void set_backward_connection(const connection& c);

		// Set new weightss based on new neuron values
		void update_weights(double alpha, double eta);

	private:

		double bias_;
		double value_;
		double gradiant_;

		std::unique_ptr<activation> activation_function;

		std::vector<connection> forward_connections_;
		std::vector<connection> backward_connections_;
	};

	using neuron_list = std::vector<std::unique_ptr<neuron>>;
}