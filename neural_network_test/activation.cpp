#include "activation.h"

#include <algorithm>
#include <cmath>

double neural_network::activation_none::f(double x) const
{
	return x;
}

double neural_network::activation_none::derivative(double x) const
{
	return x;
}

double neural_network::activation_rectlinear::f(double x) const
{
	return std::max(x, 0.0);
}

double neural_network::activation_rectlinear::derivative(double x) const
{
	return 0.0;
}

double neural_network::activation_leaky_rectlinear::f(double x) const
{
	return 0.0;
}

double neural_network::activation_leaky_rectlinear::derivative(double x) const
{
	return 0.0;
}

double neural_network::activation_sigmoid::f(double x) const
{
	return x / (1 + std::abs(x));
}

double neural_network::activation_sigmoid::derivative(double x) const
{
	double activated = f(x);
	return activated * (1 - activated);
}

double neural_network::activation_softmax::f(double x) const
{
	return 0.0;
}

double neural_network::activation_softmax::derivative(double x) const
{
	return 0.0;
}

double neural_network::activation_tanh::f(double x) const
{
	return std::tanh(x);
}

double neural_network::activation_tanh::derivative(double x) const
{
	double th_x = tanh(x);
	return 1.0 - th_x * th_x;
}

std::unique_ptr<neural_network::activation> neural_network::convert(activation_type type)
{
	switch (type)
	{
	case neural_network::activation_type::none:
		return std::unique_ptr<activation>{ new activation_none };

	case neural_network::activation_type::rectlinear:
		return std::unique_ptr<activation>{ new activation_rectlinear };

	case neural_network::activation_type::leaky_rectlinear:
		return std::unique_ptr<activation>{ new activation_leaky_rectlinear };

	case neural_network::activation_type::sigmoid:
		return std::unique_ptr<activation>{ new activation_sigmoid };

	case neural_network::activation_type::softmax:
		return std::unique_ptr<activation>{ new activation_softmax };

	case neural_network::activation_type::tanh:
		return std::unique_ptr<activation>{ new activation_tanh };

	default:
		return std::unique_ptr<activation>{ new activation_none };

	}
}
