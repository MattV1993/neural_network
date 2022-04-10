#pragma once

#include "network_spec.h"
#include "layer.h"

#include <vector>
#include <functional>
#include <exception>

namespace neural_network
{
	class network
	{
	public:

		network(const network_spec& spec);

		void train(const std::vector<double>& input, const std::vector<double>& targets);
		std::vector<double> results() const;

	private:

		void feed_forward(const std::vector<double>& input);
		void back_propigate(const std::vector<double>& targets);

		void calculate_errors(const std::vector<double>& targets);

		void set_layer_connections();

		std::vector<layer> layers;
		network_log_delegate log_delegate;
		size_t pass;

		double alpha;
		double eta;

		double error;
		double recent_average_error;
		double recent_average_smoothing_factor;
	};

	struct invalid_network : public std::exception
	{
		const char* what() const throw ()
		{
			return "Network is invalid";
		}
	};

	struct invalid_data : public std::exception
	{
		const char* what() const throw ()
		{
			return "Input data is invalid";
		}
	};
}