#pragma once

#include <memory>

namespace neural_network
{
	enum class activation_type;

	// Base class for different activation types
	// Uses stratergy pattern
	class activation
	{
	public:

		virtual ~activation() = default;

		virtual double f(double x) const = 0;
		virtual double derivative(double x) const = 0;

	protected:

		activation() = default;
		activation(const activation & other) = default;
		activation(activation && other) = default;

		activation& operator=(const activation& other) = default;
		activation& operator=(activation&& other) = default;
	};

	std::unique_ptr<activation> convert(activation_type type);

	class activation_none : public activation
	{
	public:

		double f(double x) const override;
		double derivative(double x) const override;

	};

	class activation_rectlinear : public activation
	{
	public:

		double f(double x) const override;
		double derivative(double x) const override;

	};

	class activation_leaky_rectlinear : public activation
	{
	public:

		double f(double x) const override;
		double derivative(double x) const override;

	};

	class activation_sigmoid : public activation
	{
	public:

		double f(double x) const override;
		double derivative(double x) const override;

	};

	class activation_softmax : public activation
	{
	public:

		double f(double x) const override;
		double derivative(double x) const override;

	};

	class activation_tanh : public activation
	{
	public:

		double f(double x) const override;
		double derivative(double x) const override;

	};

	enum class activation_type
	{
		none,
		rectlinear,
		leaky_rectlinear,
		sigmoid,
		softmax,
		tanh
	};

}