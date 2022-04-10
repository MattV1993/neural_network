#pragma once

#include "layer_spec.h"
#include <vector>
#include <functional>

namespace neural_network
{
	struct network_log
	{
		uint32_t pass;
		std::vector<double> inputs;
		std::vector<double> outputs;
		std::vector<double> targets;
		double recent_average_error;
	};

	using network_log_delegate = std::function<void(const network_log&)>;

	struct network_spec
	{
		std::vector<layer_spec> layer_specs;
		output_layer_spec output_spec;
		network_log_delegate log_delegate = nullptr;
		double alpha = 0.7;
		double estimated_time_of_arrival = 0.15;
	};
}