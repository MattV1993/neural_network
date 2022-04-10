#pragma once

#include <vector>
#include <functional>
#include <iostream>
#include <sstream>

template <class T>
class matrix
{
public:

	matrix();
	matrix(int height, int width);
	matrix(const std::vector<std::vector<T>>& array);

	matrix multiply(double value); // scalar multiplication

	matrix add(const matrix& m) const; // addition
	matrix subtract(const matrix& m) const; // subtraction
	matrix multiply(const matrix& m) const; // hadamard product

	matrix dot(const matrix& m) const; // dot product
	matrix transpose() const; // transposed matrix

	matrix applyFunction(std::function<double(double)> function) const;

	void print(std::ostream& flux) const;

private:

	std::vector<std::vector<T> > array;

	int height;
	int width;
};

template<class T>
std::ostream& operator<<(std::ostream& flux, matrix<T> const& m);

template<class T>
inline matrix<T>::matrix()
{
}

template<class T>
inline matrix<T>::matrix(int height, int width) :
	width(width), height(height)
{
	this->array = std::vector<std::vector<T> >(height, std::vector<T>(width));
}

template<class T>
inline matrix<T>::matrix(const std::vector<std::vector<T>>& array)
{
	this->height = array.size();
	this->width = array[0].size();
	this->array = array;
}

template<class T>
inline matrix<T> matrix<T>::multiply(double value)
{
	matrix<T> result(height, width);
	int i, j;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			result.array[i][j] = array[i][j] * value;
		}
	}

	return result;
}

template<class T>
inline matrix<T> matrix<T>::add(const matrix<T>& m) const
{
	matrix<T> result(height, width);
	int i, j;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			result.array[i][j] = array[i][j] + m.array[i][j];
		}
	}

	return result;
}

template<class T>
inline matrix<T> matrix<T>::subtract(const matrix<T>& m) const
{
	matrix<T> result(height, width);
	int i, j;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			result.array[i][j] = array[i][j] - m.array[i][j];
		}
	}

	return result;
}

template<class T>
inline matrix<T> matrix<T>::multiply(const matrix<T>& m) const
{
	matrix<T> result(height, width);

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			result.array[i][j] = array[i][j] * m.array[i][j];
		}
	}
	return result;
}

template<class T>
inline matrix<T> matrix<T>::dot(const matrix<T>& m) const
{
	int i, j, h, mwidth = m.width;
	double w = 0;

	matrix<T> result(height, mwidth);

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < mwidth; j++)
		{
			for (h = 0; h < width; h++)
			{
				w += array[i][h] * m.array[h][j];
			}
			result.array[i][j] = w;
			w = 0;
		}
	}

	return result;
}

template<class T>
inline matrix<T> matrix<T>::transpose() const
{
	matrix<T> result(width, height);
	int i, j;

	for (i = 0; i < width; i++) {
		for (j = 0; j < height; j++) {
			result.array[i][j] = array[j][i];
		}
	}
	return result;
}

template<class T>
inline matrix<T> matrix<T>::applyFunction(std::function<double(double)> function) const
{
	matrix<T> result(height, width);
	int i, j;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++) {
			result.array[i][j] = function(array[i][j]);
		}
	}

	return result;
}

template<class T>
inline void matrix<T>::print(std::ostream& flux) const
{
	int i, j;
	std::vector<int> maxLength(width);
	std::stringstream ss;

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			ss << array[i][j];
			if (maxLength[j] < ss.str().size())
			{
				maxLength[j] = ss.str().size();
			}
			ss.str(std::string());
		}
	}

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			flux << array[i][j];
			ss << array[i][j];
			for (int k = 0; k < maxLength[j] - ss.str().size() + 1; k++)
			{
				flux << " ";
			}
			ss.str(std::string());
		}
		flux << std::endl;
	}
}

template<class T>
inline std::ostream& operator<<(std::ostream& flux, matrix<T> const& m)
{
	m.print(flux);
	return flux;
}
