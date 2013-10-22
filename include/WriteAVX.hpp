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


#ifndef WRITE_AVX_HPP
#define WRITE_AVX_HPP

namespace isa {
namespace Benchmarks {

void writeAVX(float * const __restrict__ a, const long long unsigned int size);


// Implementation

void writeAVX(float * const __restrict__ a, const long long unsigned int size) {
	#pragma omp parallel for schedule(static)
	for ( long long unsigned int item = 0; item < size; item += 8 ) {
		__mm256 reg;
		_mm256_store_ps(a + item, reg);
	}
}

} // Benchmarks
} // isa

#endif // WRITE_AVX_HPP 