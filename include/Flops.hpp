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


#ifndef FLOPS_HPP
#define FLOPS_HPP

namespace isa {
namespace Benchmarks {

std::string  * getFlopsOpenCL(const unsigned int iterations, const unsigned int vector, const std::string & dataType);
std::string  * getFlopsOpenCL4x2(const unsigned int iterations, const std::string & dataType);


// Implementations
std::string * getFlopsOpenCL(const unsigned int iterations, const unsigned int vector, const std::string & dataType) {
  std::string * code = new std::string();

  if ( vector == 1 ) {
    *code = "__kernel void flops(__global const " + dataType + " * restrict const A, __global const " + dataType + " * restrict const B, __global " + dataType + " * C) {\n"
      + dataType + " a;\n"
      + dataType + " b;\n"
      + dataType + " c;\n"
      // Load
      "a = A[get_global_id(0)];\n"
      "b = B[get_global_id(0)];\n"
      // Compute
      "<%COMPUTE%>"
      // Store
      "C[get_global_id(0)] = c;\n"
      "}\n";
  } else {
    *code = "__kernel void flops(__global const " + dataType + " * restrict const A, __global const " + dataType + " * restrict const B, __global " + dataType + " * C) {\n"
      + dataType + isa::utils::toString(vector) + " a;\n"
      + dataType + isa::utils::toString(vector) + " b;\n"
      + dataType + isa::utils::toString(vector) + " c;\n"
      // Load
      "a = vload" + isa::utils::toString(vector) + "(0, &(A[get_global_id(0) * " + isa::utils::toString(vector) + "]));\n"
      "b = vload" + isa::utils::toString(vector) + "(0, &(B[get_global_id(0) * " + isa::utils::toString(vector) + "]));\n"
      // Compute
      "<%COMPUTE%>"
      // Store
      "vstore" + isa::utils::toString(vector) + "(c, 0, &(C[get_global_id(0) * " + isa::utils::toString(vector) + "]));\n"
      "}\n";
  }
  std::string computeTemplate = "c = (a + b) * get_local_id(0);\n";

  std::string * compute_s = new std::string();

  for ( unsigned int i = 0; i < iterations; i++ ) {
    compute_s->append(computeTemplate);
  }

  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);

  return code;
}

std::string * getFlopsOpenCL4x2(const unsigned int iterations, const std::string & dataType) {
  std::string * code = new std::string();

  *code = "__kernel void flops4x2(__global const " + dataType + " * restrict const A, __global const " + dataType + " * restrict const B, __global " + dataType + " * C) {\n"
    + dataType + "4 temp;\n"
    + dataType + "2 a_0;\n"
    + dataType + "2 a_1;\n"
    + dataType + "2 b_0;\n"
    + dataType + "2 b_1;\n"
    + dataType + "2 c_0;\n"
    + dataType + "2 c_1;\n"
    // Load
    "temp = vload" + "4(0, &(A[get_global_id(0) * 4]));\n"
    "a_0 = (" + dataType + "2)(temp.s0, temp.s1);\n"
    "a_1 = (" + dataType + "2)(temp.s2, temp.s3);\n"
    "temp = vload" + "4(0, &(B[get_global_id(0) * 4]));\n"
    "b_0 = (" + dataType + "2)(temp.s0, temp.s1);\n"
    "b_1 = (" + dataType + "2)(temp.s2, temp.s3);\n"
    // Compute
    "<%COMPUTE%>"
    // Store
    "temp = (" + dataType + "4)(c_0.s0, c_0.s1, c_1.s0, c_1.s1);\n"
    "vstore" + "4(temp, 0, &(C[get_global_id(0) * 4]));\n"
    "}\n";
  std::string computeTemplate = "c_0 = (a_0 + b_0) * get_local_id(0);\n"
    "c_1 = (a_1 + b_1) * get_local_id(0);\n";

  std::string * compute_s = new std::string();

  for ( unsigned int i = 0; i < iterations; i++ ) {
    compute_s->append(computeTemplate);
  }

  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);

  return code;
}
} // Benchmarks
} // isa

#endif

