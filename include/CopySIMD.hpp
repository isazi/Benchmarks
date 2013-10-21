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


#ifndef COPY_SIMD_HPP
#define COPY_SIMD_HPP

namespace isa {
namespace Benchmarks {

// SSE
void copySSE(float * const __restrict__ a, const float * const __restrict__ b, const long long unsigned int size);

// AVX
void copyAVX(float * const __restrict__ a, const float * const __restrict__ b, const long long unsigned int size);


// Implementation

void copySSE(float * const __restrict__ a, const float * const __restrict__ b, const long long unsigned int size) {
	//#pragma omp parallel for
	for ( long long unsigned int item = 0; item < size; item += 4 ) {
		_mm_store_ps(a + item, _mm_load_ps(b + item));
	}
}

void copyAVX(float * const __restrict__ a, const float * const __restrict__ b, const long long unsigned int size) {
	#pragma omp parallel for
	for ( long long unsigned int item = 0; item < size; item += 8 ) {
		_mm256_store_ps(a + item, _mm256_load_ps(b + item));
	}
}

} // Benchmarks
} // isa

#endif // COPY_SIMD_HPP 