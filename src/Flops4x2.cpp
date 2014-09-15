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

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <iomanip>
#include <ctime>

#include <ArgumentList.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Stats.hpp>
#include <Flops.hpp>

typedef float dataType;
std::string typeName("float");


int main(int argc, char * argv[]) {
  unsigned int clPlatform = 0;
  unsigned int clDevice = 0;
  unsigned int extIterations = 0;
  unsigned int intIterations = 0;
  unsigned int N = 0;
  unsigned int maxThreads = 0;
  unsigned int threadUnit = 0;
  isa::utils::ArgumentList args(argc, argv);

  try {
    clPlatform = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDevice = args.getSwitchArgument< unsigned int >("-opencl_device");
    extIterations = args.getSwitchArgument< unsigned int >("-ext_iterations");
    intIterations = args.getSwitchArgument< unsigned int >("-int_iterations");
    N = args.getSwitchArgument< unsigned int >("-N");
    maxThreads = args.getSwitchArgument< unsigned int >("-max_threads");
    threadUnit = args.getSwitchArgument< unsigned int >("-thread_unit");
  } catch ( isa::utils::EmptyCommandLine & err ) {
    std::cerr << args.getName() << " -opencl_platform ... -opencl_device ... -ext_iterations ... -int_iterations ... -N ... -max_threads ... -thread_unit ..." << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Initialize OpenCL
	cl::Context * clContext = new cl::Context();
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatform, 1, clPlatforms, clContext, clDevices, clQueues);

  // Allocate memory
  std::vector< dataType > A(N);
  std::vector< dataType > B(N);
  cl::Buffer A_d, B_d, C_d;
  try {
    A_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, A.size() * sizeof(dataType), 0, 0);
    B_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, B.size() * sizeof(dataType), 0, 0);
    C_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, A.size() * sizeof(dataType), 0, 0);
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error allocating memory: " << isa::utils::toString(err.err()) << "." << std::endl;
  }

  std::srand(std::time(0));
  for ( unsigned int n = 0; n < N; n++ ) {
    A[n] = static_cast< dataType >(std::rand() % 100);
    B[n] = A[n];
  }

  // Copy data structures to device
  try {
    clQueues->at(clDevice)[0].enqueueWriteBuffer(A_d, CL_FALSE, 0, A.size() * sizeof(dataType), reinterpret_cast< void * >(A.data()), 0, 0);
    clQueues->at(clDevice)[0].enqueueWriteBuffer(B_d, CL_FALSE, 0, B.size() * sizeof(dataType), reinterpret_cast< void * >(B.data()), 0, 0);
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString(err.err()) << "." << std::endl;
    return 1;
  }

  for ( unsigned int threads = threadUnit; threads <= maxThreads; threads += threadUnit ) {
    if ( (N % threads) != 0 ) {
      continue;
    }
    cl::Kernel * kernel;
    isa::utils::Timer kernelTimer("Kernel Timer");
    isa::utils::Stats< double > kernelGFlops;
    isa::utils::Stats< double > kernelGBs;
    std::string * code = isa::Benchmarks::getFlopsOpenCL4x2(intIterations, typeName);
    
    try {
      kernel = isa::OpenCL::compile("flops4x2", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDevice));
    } catch ( isa::OpenCL::OpenCLError & err ) {
      std::cerr << err.what() << std::endl;
      return 1;
    }

    try {
      double flops = isa::utils::giga(static_cast< long long unsigned int >(N) * 2 * iterations);
      double gbs = isa::utils::giga(static_cast< long long unsigned int >(N) * 3 * sizeof(dataType));
      cl::Event kernelSync;
      cl::NDRange global(N / 4);
      cl::NDRange local(threads);
      
      kernel->setArg(0, A_d);
      kernel->setArg(1, B_d);
      kernel->setArg(2, C_d);
      // Warm-up
      clQueues->at(clDevice)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &kernelSync);
      kernelSync.wait();
      for ( unsigned int iteration = 0; iteration < extIterations; iteration++ ) {
        kernelTimer.start();
        clQueues->at(clDevice)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &kernelSync);
        kernelSync.wait();
        kernelTimer.stop();
        kernelGFlops.addElement(flops / kernelTimer.getLastRunTime());
        kernelGBs.addElement(gbs / kernelTimer.getLastRunTime());
      }
    } catch ( cl::Error & err ) {
      std::cerr << "OpenCL error kernel execution: " << isa::utils::toString(err.err()) << "." << std::endl;
    }

    std::cout << std::fixed;
    std::cout << threads << " " << vector << " ";
    std::cout << std::setprecision(3);
    std::cout << kernelGFlops.getAverage() << " " << kernelGFlops.getStdDev() << " ";
    std::cout << kernelGBs.getAverage() << " " << kernelGBs.getStdDev() << " ";
    std::cout << std::setprecision(6);
    std::cout << kernelTimer.getAverageTime() << " " << kernelTimer.getStdDev();
    std::cout << std::endl;
  }

  return 0;
}

