// Implements the KPM method. Calculates the DOS of a linear chain
// https ://arxiv.org/pdf/cond-mat/0504627.pdf



#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <complex>
#include <vector>
#include <cstdlib>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace
{
	typedef float Real;
	typedef std::complex<Real> Scalar;

	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
	typedef Eigen::SparseMatrix<Scalar> SMatrix;
	typedef Eigen::Triplet<Scalar> Triplet;
	typedef Eigen::Array<Real, Eigen::Dynamic, 1> Array;

	const Real PI = (Real)EIGEN_PI;
}


SMatrix hamiltonian_chain(const int dimension=100, Scalar t=-1.);
inline Real jackson_kernel(int moment, const int num_moments);
inline Real chebysev_polynomials(int moments, Real energy);
Array moments(const SMatrix h, const int num_moments, Vector alpha, Vector beta);
inline Real kernel(int moment, const int num_moments, std::string kernel_type);
Array modified_moments(const SMatrix h, const int num_moments, std::string kernel_type);
Real energy_expansion(Real energy, Array mu, const int num_moments);
Array calculate_dos(Array energy, Array mu, const int num_moments);


int main()
{
	std::srand((unsigned int) 42);  // seed of random
	const int dimension = 1'000;  // size of the system
	Scalar t = -0.4;
	const SMatrix hamiltonian = hamiltonian_chain(dimension, t);
	const Array energy = Array::LinSpaced(100, -0.9, 0.9);

	//some rescaling  needs to be done

	// calculating the moments
	const int num_moments = 100;  // should be even or else seg faults in calculate_moments
	Array mu = modified_moments(hamiltonian, num_moments, "jackson");

	// calculate the dos
	Array dos = calculate_dos(energy, mu, num_moments);

	// Just spits out the DOS
	std::cout << dos;

}


SMatrix hamiltonian_chain(const int dimension, Scalar t)
{
	SMatrix mat(dimension, dimension);

	const int estimation_of_entries = 2 * dimension;
	std::vector<Triplet> tripletList;
	tripletList.reserve(estimation_of_entries);

	for (int i = 0; i < dimension - 1; ++i)
	{
		tripletList.push_back(Triplet(i, i + 1, t));
		tripletList.push_back(Triplet(i + 1, i, t));
	}
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	return mat;
}


inline Real jackson_kernel(int moment, const int num_moments)
{
	return (
		(num_moments - moment + 1) * std::cos(PI * moment / (num_moments + 1)) +
		std::sin(PI * moment / (num_moments + 1)) / std::tan(PI / (num_moments + 1))
		) /
		(num_moments + 1);
}


inline Real kernel(int moment, const int num_moments, std::string kernel_type)
{
	if (kernel_type == "jackson")
	{
		return jackson_kernel(moment, num_moments);
	}
	else
	{
		std::cout << "No kernel specified! Using Dirichlet kernel\n";
		return 1.;
	}
}


inline Real chebysev_polynomials(int moment, Real energy)
{
	return std::cos(moment * std::acos(energy));
}


Array moments(const SMatrix h, const int num_moments, Vector alpha, Vector beta)
{
	Array mu = Array::Zero(num_moments);
	Vector r0 = alpha;
	Vector r2 = beta;
	Vector r1 = r0;

	mu(0) = 1.;
	if (0 < num_moments)
	{
		mu(1) = (r2.adjoint() * r0).value().real();
	}
	for (int i = 1; i < num_moments / 2; ++i)
	{
		mu(2 * i) = (2 * r2.adjoint() * r2).value().real() - mu(0);
		r1 = 2 * h * r2 - r0;  // rn -> rn+1
		r0 = r2;
		r2 = r1;
		mu(2 * i + 1) = (2 * r2.adjoint() * r0).value().real() - mu(1);
	}
	return mu;
}


Array modified_moments(const SMatrix h, const int num_moments, std::string kernel_type)
{
	Vector alpha = Vector::Random(h.rows()).normalized();  // random vector, range: ]-1, 1[
	Vector beta = h * alpha;
	Array mu = moments(h, num_moments, alpha, beta);

	for (int i = 0; i < mu.size(); ++i)
	{
		mu(i) *= kernel(i, num_moments, kernel_type);
	}
	return mu;
}


Real energy_expansion(Real energy, Array mu, const int num_moments)
//expands energy in terms of Chebysev polynomials
{
	Real energy_expanded = 0;
	energy_expanded += chebysev_polynomials(0, energy) * mu(0);

	for (int i = 1; i < num_moments; ++i)
	{
		energy_expanded += 2 * chebysev_polynomials(i, energy) * mu(i);
	}
	return energy_expanded;
}


Array calculate_dos(Array energy, Array mu, const int num_moments)
{
	Array dos = Array::Zero(energy.size());
	Real ei;
	Real denominator;
	Real numerator;
	for (int i = 0; i < energy.size(); ++i)
	{
		ei = energy(i);
		denominator = PI * std::sqrt(1.0 - ei * ei);  // PI * sqrt(1-x^2)
		numerator = 2 * energy_expansion(ei, mu, num_moments);
		dos(i) = numerator / denominator;
	}

	return dos;
}
