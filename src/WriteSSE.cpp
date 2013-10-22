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
#include <WriteSSE.hpp>
#include <Timer.hpp>
#include <utils.hpp>
using isa::utils::ArgumentList;
using isa::Benchmarks::writeSSE;
using isa::utils::Timer;
using isa::utils::same;
using isa::utils::giga;

const unsigned int nrIterations = 10;


int main(int argc, char * argv[]) {
	unsigned int arrayDim = 0;

	// Parse command line
	if ( argc != 3 ) {
		cerr << "Usage: " << argv[0] << " -n <array_size>" << endl;
		return 1;
	}

	ArgumentList commandLine(argc, argv);
	try {
		arrayDim = commandLine.getSwitchArgument< long long unsigned int >("-n");
	} catch ( exception & err ) {
		cerr << err.what() << endl;
		return 1;
	}

	Timer copyTimer = Timer("writeSSE");
	float * A = new float [arrayDim];

	writeSSE(A, arrayDim);
	for ( unsigned int iter = 0; iter < nrIterations; iter++ ) {
		copyTimer.start();
		writeSSE(A, arrayDim);
		copyTimer.stop();
	}

	cout << fixed << setprecision(3) << giga(arrayDim * sizeof(float)) / copyTimer.getAverageTime() << endl;

	return 0;
}
