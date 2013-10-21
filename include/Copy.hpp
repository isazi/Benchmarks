/*
 * Copyright (C) 2013
 * Alessio Sclocco <a.sclocco@vu.nl>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <x86intrin.h>
#include <omp.h>

#include <kernels/Kernel.hpp>
#include <CLData.hpp>
#include <Exceptions.hpp>
#include <utils.hpp>
using isa::OpenCL::Kernel;
using isa::OpenCL::CLData;
using isa::Exceptions::OpenCLError;
using isa::utils::giga;


#ifndef COPY_HPP
#define COPY_HPP

namespace isa {

namespace Benchmarks {

template < typename T > class Copy : public Kernel< T > {
public:
	Copy(string dataType);

	void generateCode() throw (OpenCLError);
	void operator()(CLData< T > * a, CLData< T > * b) throw (OpenCLError);

	inline void setNrThreadsPerBlock(unsigned int threads);
	inline void setNrThreads(unsigned int threads);
	inline void setNrRows(unsigned int rows);

private:
	unsigned int nrThreadsPerBlock;
	unsigned int nrThreads;
	unsigned int nrRows;
};


// Implementation

template< typename T > Copy< T >::Copy(string dataType) : Kernel< T >("Copy", dataType), nrThreadsPerBlock(0), nrThreads(0), nrRows(0) {}


template< typename T > void Copy< T >::generateCode() throw (OpenCLError) {
	this->gb = giga(static_cast< long long unsigned int >(nrThreads) * 2 * sizeof(T));
	
	delete this->code;
	this->code = new string();
	*(this->code) = "__kernel void " + this->name + "(__global " + this->dataType + " * const restrict A, __global const " +  this->dataType + " * const restrict B) {\n"
		"const unsigned int id = ( get_global_id(1) * get_global_size(0) ) + get_global_id(0);\n"
		"A[id] = B[id];\n"
		"}";

	this->compile();
}


template< typename T > void Copy< T >::operator()(CLData< T > * a, CLData< T > * b) throw (OpenCLError) {
	cl::NDRange globalSize(nrThreads / nrRows, nrRows);
	cl::NDRange localSize(nrThreadsPerBlock, 1);

	this->setArgument(0, *(a->getDeviceData()));
	this->setArgument(1, *(b->getDeviceData()));

	this->run(globalSize, localSize);
}


template< typename T > inline void Copy< T >::setNrThreadsPerBlock(unsigned int threads) {
	nrThreadsPerBlock = threads;
}


template< typename T > inline void Copy< T >::setNrThreads(unsigned int threads) {
	nrThreads = threads;
}


template< typename T > inline void Copy< T >::setNrRows(unsigned int rows) {
	nrRows = rows;
}

} // Benchmarks
} // isa

#endif // COPY_HPP
