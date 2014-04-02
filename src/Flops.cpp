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

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
#include <iomanip>
using std::setprecision;
#include <ctime>

#include <ArgumentList.hpp>
using isa::utils::ArgumentList;
#include <InitializeOpenCL.hpp>
using isa::OpenCL::initializeOpenCL;
#include <CLData.hpp>
using isa::OpenCL::CLData;
#include <Exceptions.hpp>
using isa::Exceptions::OpenCLError;
#include <Flops.hpp>
using isa::Benchmarks::Flops;
#include <utils.hpp>
using isa::utils::same;

const unsigned int nrIterations = 10;


int main(int argc, char * argv[]) {
	unsigned int oclPlatform = 0;
	unsigned int oclDevice = 0;
	unsigned int arrayDim = 0;
	unsigned int nrLoops = 0;
	unsigned int minThreads = 0;
	unsigned int maxThreads = 0;

	// Parse command line
	if ( argc != 11 ) {
		cerr << "Usage: " << argv[0] << " -opencl_platform <opencl_platform> -opencl_device <opencl_device> -loops <nr_loops> -min <min_threads> -max <max_threads>" << endl;
		return 1;
	}

	ArgumentList commandLine(argc, argv);
	try {
		oclPlatform = commandLine.getSwitchArgument< unsigned int >("-opencl_platform");
		oclDevice = commandLine.getSwitchArgument< unsigned int >("-opencl_device");
		nrLoops = commandLine.getSwitchArgument< unsigned int >("-loops");
		minThreads = commandLine.getSwitchArgument< unsigned int >("-min");
		maxThreads = commandLine.getSwitchArgument< unsigned int >("-max");
	} catch ( exception & err ) {
		cerr << err.what() << endl;
		return 1;
	}

	// Initialize OpenCL
	vector< cl::Platform > * oclPlatforms = new vector< cl::Platform >();
	cl::Context * oclContext = new cl::Context();
	vector< cl::Device > * oclDevices = new vector< cl::Device >();
	vector< vector< cl::CommandQueue > > * oclQueues = new vector< vector< cl::CommandQueue > >();
	try {
		initializeOpenCL(oclPlatform, 1, oclPlatforms, oclContext, oclDevices, oclQueues);
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}
	arrayDim = (oclDevices->at(oclDevice)).getInfo< CL_DEVICE_MAX_MEM_ALLOC_SIZE >();
	arrayDim /= sizeof(float);

	CLData< float > * A = new CLData< float >("A", true);
	CLData< float > * C = new CLData< float >("C", true);

	A->setCLContext(oclContext);
	A->setCLQueue(&(oclQueues->at(oclDevice)[0]));
	A->allocateHostData(arrayDim);
	srand(time(NULL));
	for ( unsigned int i = 0; i < arrayDim; i++ ) {
		A->setHostDataItem(i, 1.0f / static_cast< float >(rand() % 15));
	}
	C->setCLContext(oclContext);
	C->setCLQueue(&(oclQueues->at(oclDevice)[0]));
	C->allocateHostData(arrayDim);
	try {
		A->setDeviceReadOnly();
		A->allocateDeviceData();
		A->copyHostToDevice();
		C->setDeviceWriteOnly();
		C->allocateDeviceData();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	cout << fixed << setprecision(3) << endl;
	for (unsigned int threads0 = minThreads; threads0 <= maxThreads; threads0 *= 2 ) {
		for (unsigned int threads1 = 1; threads1 <= 32; threads1++ ) {
			if ( (arrayDim % (threads0 * threads1) != 0) || ((threads0 * threads1) > maxThreads) ) {
				continue;
			}

			Flops< float > flops = Flops< float >("float");
			try {
				flops.bindOpenCL(oclContext, &(oclDevices->at(oclDevice)), &(oclQueues->at(oclDevice)[0]));
				flops.setNrThreads(arrayDim);
				flops.setNrThreadsPerBlock(threads0);
				flops.setNrRows(threads1);
				flops.setNrIterations(nrLoops);
				flops.generateCode();

				flops(A, C);
				(flops.getTimer()).reset();
				for ( unsigned int iter = 0; iter < nrIterations; iter++ ) {
					flops(A, C);
				}
			} catch ( OpenCLError err ) {
				cerr << err.what() << endl;
				return 1;
			}

			cout << threads0 << " " << threads1 << " " << flops.getGFLOPs() << " " << flops.getGBs() << endl;

		}
	}
	cout << endl;

	return 0;
}
