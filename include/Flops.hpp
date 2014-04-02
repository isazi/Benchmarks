//
// Copyright (C) 2014
// Alessio Sclocco <a.sclocco@vu.nl>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#include <Kernel.hpp>
using isa::OpenCL::Kernel;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <Exceptions.hpp>
using isa::Exceptions::OpenCLError;
#include <utils.hpp>
using isa::utils::giga;
using isa::utils::toStringValue;


#ifndef FLOPS_HPP
#define FLOPS_HPP

namespace isa {

namespace Benchmarks {

template < typename T > class Flops : public Kernel< T > {
public:
	Flops(string dataType);

	void generateCode() throw (OpenCLError);
	void operator()(CLData< T > * a, CLData< T > * c) throw (OpenCLError);

	inline void setNrThreadsPerBlock(unsigned int threads);
	inline void setNrThreads(unsigned int threads);
	inline void setNrRows(unsigned int rows);
	inline void setNrIterations(unsigned int iterations);

private:
	unsigned int nrThreadsPerBlock;
	unsigned int nrThreads;
	unsigned int nrRows;
	unsigned int nrIterations;
};


// Implementation

template< typename T > Flops< T >::Flops(string dataType) : Kernel< T >("Flops", dataType), nrThreadsPerBlock(0), nrThreads(0), nrRows(0), nrIterations(0) {}


template< typename T > void Flops< T >::generateCode() throw (OpenCLError) {
	long long unsigned int ops = static_cast< long long unsigned int >(nrThreads * nrIterations * 2);
	long long unsigned int memOps = nrThreads * 2 * sizeof(T);

	this->arInt = ops / static_cast< double >(memOps);
	this->gflop = giga(ops);
	this->gb = giga(memOps);

	delete this->code;
	this->code = new string();
	*(this->code) = "__kernel void " + this->name + "(__global const " + this->dataType + " * const restrict A, __global " + this->dataType + " * const restrict C) {\n"
		"const unsigned int id = (get_global_id(1) * get_global_size(0)) + get_global_id(0);\n"
		+ this->dataType + " a = A[id];\n"
		+ this->dataType + " total = 0;\n"
		"for ( unsigned int iteration = 0; iteration < " + toStringValue< unsigned int >(nrIterations) + "; iteration++ ) {\n"
		"total += (a * get_local_id(0));\n"
		"}\n"
		"C[id] = total;\n"
		"}";

	this->compile();
}


template< typename T > void Flops< T >::operator()(CLData< T > * a, CLData< T > * c) throw (OpenCLError) {
	cl::NDRange globalSize(nrThreads / nrRows, nrRows);
	cl::NDRange localSize(nrThreadsPerBlock, 1);

	this->setArgument(0, *(a->getDeviceData()));
	this->setArgument(1, *(c->getDeviceData()));

	this->run(globalSize, localSize);
}


template< typename T > inline void Flops< T >::setNrThreadsPerBlock(unsigned int threads) {
	nrThreadsPerBlock = threads;
}


template< typename T > inline void Flops< T >::setNrThreads(unsigned int threads) {
	nrThreads = threads;
}


template< typename T > inline void Flops< T >::setNrRows(unsigned int rows) {
	nrRows = rows;
}

template< typename T > inline void Flops< T >::setNrIterations(unsigned int iteration) {
	nrIterations = iteration;
}

} // Benchmarks
} // isa

#endif // FLOPS_HPP
