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
#include <Read.hpp>
#include <utils.hpp>
using isa::utils::ArgumentList;
using isa::OpenCL::initializeOpenCL;
using isa::OpenCL::CLData;
using isa::Exceptions::OpenCLError;
using isa::Benchmarks::Read;
using isa::utils::same;

const unsigned int nrIterations = 10;


int main(int argc, char * argv[]) {
	unsigned int oclPlatform = 0;
	unsigned int oclDevice = 0;
	unsigned int arrayDim = 0;
	unsigned int nrThreadsPerBlock = 0;
	unsigned int nrRows = 0;

	// Parse command line
	if ( argc != 11 ) {
		cerr << "Usage: " << argv[0] << " -opencl_platform <opencl_platform> -opencl_device <opencl_device> -n <dim> -t <threads_per_block> -r <rows>" << endl;
		return 1;
	}

	ArgumentList commandLine(argc, argv);
	try {
		oclPlatform = commandLine.getSwitchArgument< unsigned int >("-opencl_platform");
		oclDevice = commandLine.getSwitchArgument< unsigned int >("-opencl_device");
		arrayDim = commandLine.getSwitchArgument< unsigned int >("-n");
		nrThreadsPerBlock = commandLine.getSwitchArgument< unsigned int >("-t");
		nrRows = commandLine.getSwitchArgument< unsigned int >("-r");
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

	CLData< float > * B = new CLData< float >("B", true);

	B->setCLContext(oclContext);
	B->setCLQueue(&(oclQueues->at(oclDevice)[0]));
	B->allocateHostData(arrayDim);
	try {
		B->setDeviceReadOnly();
		B->allocateDeviceData();
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	Read< float > read = Read< float >("float");
	try {
		read.bindOpenCL(oclContext, &(oclDevices->at(oclDevice)), &(oclQueues->at(oclDevice)[0]));
		read.setNrThreadsPerBlock(nrThreadsPerBlock);
		read.setNrThreads(arrayDim);
		read.setNrRows(nrRows);
		read.generateCode();

		B->copyHostToDevice(true);
		read(B);
		(read.getTimer()).reset();
		for ( unsigned int iter = 0; iter < nrIterations; iter++ ) {
			read(B);
		}
	} catch ( OpenCLError err ) {
		cerr << err.what() << endl;
		return 1;
	}

	cout << endl;
	cout << fixed << setprecision(3) << read.getGB() / (read.getTimer()).getAverageTime() << endl;
	cout << endl;

	return 0;
}
