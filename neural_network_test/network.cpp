#include "network.h"

neural_network::network::network(const network_spec& spec)
	: log_delegate{ spec.log_delegate }, pass{ 0 },
	alpha{ spec.alpha },
	eta{spec.estimated_time_of_arrival},
	error{ 0.0 }, recent_average_error{ 0.0 },
	recent_average_smoothing_factor{ 1.0 }
{
	if (spec.layer_specs.size() == 0)
	{
		throw invalid_network{};
	}

	for (const auto& spec : spec.layer_specs)
	{
		layers.emplace_back(spec);
	}

	layers.emplace_back(layer_spec{ spec.output_spec });

	set_layer_connections();
}

void neural_network::network::train(const std::vector<double>& input, const std::vector<double>& targets)
{
	if (input.size() != layers.front().neurons().size())
	{
		throw invalid_data{};
	}
	else if (targets.size() != layers.back().neurons().size())
	{
		throw invalid_data{};
	}

	feed_forward(input);
	calculate_errors(targets);
	back_propigate(targets);

	pass++;

	if (log_delegate)
	{
		network_log log;
		log.pass = pass;
		log.inputs = input;
		log.outputs = results();
		log.targets = targets;
		log.recent_average_error = recent_average_error;

		log_delegate(log);
	}
}

std::vector<double> neural_network::network::results() const
{
	std::vector<double> ret;

	for (const auto& n : layers.back().neurons())
	{
		ret.push_back(n->value());
	}

	return ret;
}

void neural_network::network::feed_forward(const std::vector<double>& input)
{
	layers.front().set_values(input);

	for (auto layer = layers.begin() + 1; layer != layers.end(); ++layer)
	{
		layer->feed_forward();
	}
}

void neural_network::network::back_propigate(const std::vector<double>& targets)
{
	layers.back().calculate_gradiant(targets);

	for (auto l = layers.rbegin() + 1; l != layers.rend() - 1; ++l)
	{
		l->calculate_gradiant();
	}

	for (auto l = layers.rbegin() + 1; l != layers.rend() - 1; ++l)
	{
		l->update_weights(alpha, eta);
	}
}

void neural_network::network::calculate_errors(const std::vector<double>& targets)
{
	const neuron_list& output_neurons = layers.back().neurons();
	error = 0.0;

	for (size_t i = 0; i < output_neurons.size(); i++)
	{
		double delta = targets[i] - output_neurons[i]->value();
		error += delta * delta;
	}

	error /= output_neurons.size();
	error = sqrt(error);

	recent_average_error =
		(recent_average_error * recent_average_smoothing_factor + error)
		/ (recent_average_smoothing_factor + 1.0);
}

void neural_network::network::set_layer_connections()
{
	for (auto l = layers.begin(); l != layers.end() - 1; ++l)
	{
		l->set_connections(*(l + 1));
	}
}