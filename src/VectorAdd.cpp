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

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cstring>
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;
using std::ofstream;
using std::ceil;
using std::pow;

#include <ArgumentList.hpp>
#include <InitializeOpenCL.hpp>
#include <CLData.hpp>
#include <Exceptions.hpp>
#include <VectorAdd.hpp>
#include <utils.hpp>
using isa::utils::ArgumentList;
using isa::OpenCL::initializeOpenCL;
using isa::OpenCL::CLData;
using isa::Exceptions::OpenCLError;
using isa::Benchmarks::VectorAdd;
using isa::utils::same;

const unsigned int nrIterations = 10;


int main(int argc, char * argv[]) {
	unsigned int oclPlatform = 0;
	unsigned int oclDevice = 0;
	unsigned int arrayDim = 0;
	unsigned int maxThreads = 0;

	// Parse command line
	if ( argc != 7 ) {
		cerr << "Usage: " << argv[0] << " -opencl_platform <opencl_platform> -opencl_device <opencl_device> -m <max_threads>" << endl;
		return 1;
	}

	ArgumentList commandLine(argc, argv);
	try {
		oclPlatform = commandLine.getSwitchArgument< unsigned int >("-opencl_platform");
		oclDevice = commandLine.getSwitchArgument< unsigned int >("-opencl_device");
		maxThreads = commandLine.getSwitchArgument< unsigned int >("-max_threads");
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
	CLData< float > * B = new CLData< float >("B", true);
	CLData< float > * C = new CLData< float >("C", true);

	A->setCLContext(oclContext);
	A->setCLQueue(&(oclQueues->at(oclDevice)[0]));
	A->allocateHostData(arrayDim);
	B->setCLContext(oclContext);
	B->setCLQueue(&(oclQueues->at(oclDevice)[0]));
	B->allocateHostData(arrayDim);
	C->setCLContext(oclContext);
	C->setCLQueue(&(oclQueues->at(oclDevice)[0]));
	C->allocateHostData(arrayDim);
	try {
		A->setDeviceReadOnly();
		A->allocateDeviceData();
		B->setDeviceReadOnly();
		B->allocateDeviceData();
		C->setDeviceWriteOnly();
		C->allocateDeviceData();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	cout << fixed << setprecision(3) << endl;
	for (unsigned int threads0 = 2; threads0 <= maxThreads; threads0 *= 2 ) {
		for (unsigned int threads1 = 1; threads1 <= 32; threads1++ ) {
			if ( (arrayDim % (threads0 * threads1) != 0) || ((threads0 * threads1) > maxThreads) ) {
				continue;
			}
			
			VectorAdd< float > vectorAdd = VectorAdd< float >("float");
			try {
				vectorAdd.bindOpenCL(oclContext, &(oclDevices->at(oclDevice)), &(oclQueues->at(oclDevice)[0]));
				vectorAdd.setNrThreads(arrayDim);
				vectorAdd.setNrThreadsPerBlock(threads0);				
				vectorAdd.setNrRows(threads1);
				vectorAdd.generateCode();

				A->copyHostToDevice(true);
				B->copyHostToDevice(true);
				vectorAdd(A, B, C);
				(vectorAdd.getTimer()).reset();
				for ( unsigned int iter = 0; iter < nrIterations; iter++ ) {
					vectorAdd(A, B, C);
				}
			} catch ( OpenCLError err ) {
				cerr << err.what() << endl;
				return 1;
			}

			cout << threads0 << " " << threads1 << " " << vectorAdd.getGFLOP() / (vectorAdd.getTimer()).getAverageTime() << " " << vectorAdd.getGB() / (vectorAdd.getTimer()).getAverageTime() << endl;
			
		}
	}
	cout << endl;

	return 0;
}