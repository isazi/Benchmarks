// Copyright 2014 Alessio Sclocco <a.sclocco@vu.nl>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include <utils.hpp>


#ifndef BANDWIDTH_HPP
#define BANDWIDTH_HPP

namespace isa {
namespace Benchmarks {

std::string  * getBandwidthOpenCL(const unsigned int vector, const std::string & dataType);


// Implementations
std::string * getBandwidthOpenCL(const unsigned int vector, const std::string & dataType) {
  std::string * code = new std::string();

  if ( vector == 1 ) {
    *code = "__kernel void bandwidth(__global const " + dataType + " * restrict const A, __global const " + dataType + " * restrict const B, __global " + dataType + " * C) {\n"
      + dataType + " a;\n"
      + dataType + " b;\n"
      + dataType + " c;\n"
      // Load
      "a = A[get_global_id(0)];\n"
      "b = B[get_global_id(0)];\n"
      // Store
      "C[get_global_id(0)] = c;\n"
      "}\n";
  } else {
    *code = "__kernel void bandwidth(__global const " + dataType + " * restrict const A, __global const " + dataType + " * restrict const B, __global " + dataType + " * C) {\n"
      + dataType + isa::utils::toString(vector) + " a;\n"
      + dataType + isa::utils::toString(vector) + " b;\n"
      + dataType + isa::utils::toString(vector) + " c;\n"
      // Load
      "a = vload" + isa::utils::toString(vector) + "(0, &(A[get_global_id(0) * " + isa::utils::toString(vector) + "]));\n"
      "b = vload" + isa::utils::toString(vector) + "(0, &(B[get_global_id(0) * " + isa::utils::toString(vector) + "]));\n"
      // Store
      "vstore" + isa::utils::toString(vector) + "(c, 0, &(C[get_global_id(0) * " + isa::utils::toString(vector) + "]));\n"
      "}\n";
  }

  return code;
}

} // Benchmarks
} // isa

#endif

